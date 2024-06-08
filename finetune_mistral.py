import json
from torch import bfloat16
import logging
import os
import datasets
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
logger = logging.getLogger(__name__)


class FineTuner:
    def __init__(self, args):
        self.train_data_path = args["train_data"]
        self.val_data_path = args["val_data"]
        self.base_mistral_model = args["base_mistral_model"]
        self.output_dir = args["output_dir"]
        self.model_max_length = args["model_max_length"]
        self.warmup_steps = args["warmup_steps"]
        self.max_steps = args["max_steps"]
        self.learning_rate = args["learning_rate"]
        self.do_eval = args["do_eval"]
        self.optimizer = 'paged_adamw_8bit'
        self.setup_accelerator()
        self.setup_datasets()
        self.setup_model()
        self.apply_peft()

        self.quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=bfloat16,
        )

    def setup_accelerator(self):
        os.environ['WANDB_DISABLED'] = 'true'
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    def setup_datasets(self):
        train_path = self.train_data_path
        val_path = self.val_data_path

        # use dirname to get /valohai/inputs/train_data from '/valohai/inputs/train_data/train.csv'
        self.tokenized_train_dataset = datasets.load_from_disk(os.path.dirname(train_path))
        self.tokenized_eval_dataset = datasets.load_from_disk(os.path.dirname(val_path))

    def setup_model(self):
        base_model_id = self.base_mistral_model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=self.quantization_config,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_id,
            model_max_length=self.model_max_length,
            padding_side='left',
            add_eos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.gradient_checkpointing_enable()

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}',
        )

    def apply_peft(self):
        model = prepare_model_for_kbit_training(self.model)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
                'lm_head',
            ],
            bias='none',
            lora_dropout=0.05,  # Conventional
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, config)

        self.print_trainable_parameters()

        self.model = self.accelerator.prepare_model(model)

    def train(self):
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=self.output_dir,
                warmup_steps=self.warmup_steps,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,  # Want about 10x smaller than the Mistral learning rate
                logging_steps=10,
                bf16=False,
                tf32=False,
                optim=self.optimizer,
                logging_dir='./logs',  # Directory for storing logs
                save_strategy='steps',  # Save the model checkpoint every logging step
                save_steps=10,  # Save checkpoints every 10 steps
                evaluation_strategy='steps',  # Evaluate the model every logging step
                eval_steps=17,  # Evaluate and save checkpoints every 17 steps
                do_eval=self.do_eval,  # Perform evaluation at the end of training
                report_to=None,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[PrinterCallback],
        )

        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        model_save_dir = os.path.join(self.output_dir, 'best_model')

        trainer.save_model(model_save_dir)

        # save metadata
        self.save_metadata(model_save_dir)

    @staticmethod
    def save_metadata(save_dir):
        project_name = "mistral"

        metadata = {
            'dataset-versions': [
                {
                    'uri': f'dataset://mistral-models/{project_name}',
                    'targeting_aliases': ['best_mistral_checkpoint'],
                    'tags': ['dev', 'mistral'],
                },
            ],
        }
        for file in os.listdir(save_dir):
            md_path = os.path.join(save_dir, f'{file}.metadata.json')
            metadata_path = md_path
            with open(metadata_path, 'w') as outfile:
                json.dump(metadata, outfile)


class PrinterCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop('total_flos', None)
        print(json.dumps(logs))


def main():
    logging.basicConfig(level=logging.INFO)

    # Add arguments based on your script's needs
    # fmt: off

    cwd = os.getcwd()
    args = {
        "base_mistral_model": "mistralai/Mistral-7B-v0.3",
        "train_data": f"{cwd}_encoded_train",
        "val_data": f"{cwd}_encoded_test",
        "output_dir": f"{cwd}_fine_tuned_model",
        "model_max_length": 512,
        "warmup_steps": 5,
        "max_steps": 10,
        "learning_rate": 2.5e-5,
        "do_eval": "store_true"
    }
    # fmt: on

    fine_tuner = FineTuner(args)
    fine_tuner.train()


if __name__ == '__main__':
    main()
