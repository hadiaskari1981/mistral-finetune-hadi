import os
from datetime import datetime
import transformers
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
# peft parameters efficient fine-tuning
import logging
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)


class BaseModelLoader:
    def __init__(self, args):
        self.base_model_name = args["base_model_name"]
        self.cache_dir = args.get("cache_dir", None)
        self.is_inference = args.get("is_inference")
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if self.is_inference:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=self.quantization_config,
                cache_dir=self.cache_dir,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=self.quantization_config,
                cache_dir=self.cache_dir)


class DataTokenizer:
    def __init__(self, args: dict):
        self.data_path = args.get("data_path", "gem/viggo")
        self.model_max_length = args["model_max_length"]
        self.base_model_name = args["base_model_name"]
        self.cache_dir = args.get("cache_dir", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True,
            cache_dir=self.cache_dir)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_dataset = load_dataset(
            self.data_path, split='train', trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.val_dataset = load_dataset(
            self.data_path, split='validation', trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.test_dataset = load_dataset(
            self.data_path, split='test', trust_remote_code=True, cache_dir=self.cache_dir
        )

        self.tokenized_train_dataset = self.train_dataset.map(self.generate_and_tokenize_prompt)
        self.tokenized_val_dataset = self.val_dataset.map(self.generate_and_tokenize_prompt)

    def prompt_tokenizer(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
                    This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
                    The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']
                
                    ### Target sentence:
                    {data_point["target"]}
                
                    ### Meaning representation:
                    {data_point["meaning_representation"]}
                    """
        return self.prompt_tokenizer(full_prompt)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def accelerator():
    plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    acc = Accelerator(fsdp_plugin=plugin)

    return acc


def fine_tune(args):
    base_model = BaseModelLoader(args).model

    data_tokenizer = DataTokenizer(args)

    tokenized_train_dataset = data_tokenizer.tokenized_train_dataset
    tokenized_val_dataset = data_tokenizer.tokenized_val_dataset

    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    base_model = get_peft_model(base_model, CONFIG)
    print_trainable_parameters(base_model)

    # Apply the accelerator. You can comment this out to remove the accelerator.
    acc = accelerator()
    base_model = acc.prepare_model(base_model)

    tokenizer = data_tokenizer.tokenizer

    trainer = transformers.Trainer(
        model=base_model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=args.get("model_output_dir"),
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=100,
            learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
            logging_steps=50,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir=args.get("logs_dir"),  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=10,  # Save checkpoints every 50 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=10,  # Evaluate and save checkpoints every 50 steps
            do_eval=True,  # Perform evaluation at the end of training
            run_name=f"{args.get('project_name')}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return trainer


def base_model_inference(args):
    eval_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
       This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
       The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

       ### Target sentence:
       {args.get("target_sentence")}

       ### Meaning representation:
       """

    # Re-init the tokenizer so it doesn't add padding or eos token
    eval_tokenizer = AutoTokenizer.from_pretrained(
        args.get("base_model_name"),
        add_bos_token=True,
    )
    base_model = BaseModelLoader(args).model
    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    base_model.eval()
    with torch.no_grad():
        print(
            eval_tokenizer.decode(base_model.generate(**model_input, max_new_tokens=256)[0], skip_special_tokens=True))


def inference(inference_args):

    base_model = BaseModelLoader(inference_args).model

    base_model_name = inference_args.get("base_model_name")
    cache_dir = inference_args.get("cache_dir")
    output_dir = inference_args.get("model_output_dir")
    target_sentence = inference_args.get("target_sentence")
    checkpoint_version = inference_args.get("checkpoint")

    eval_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        add_bos_token=True,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    ft_model = PeftModel.from_pretrained(base_model, f"{output_dir}/{checkpoint_version}")
    eval_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
    This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
    The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

    ### Target sentence:
    {target_sentence}

    ### Meaning representation:
    """

    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad():
        print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=256)[0], skip_special_tokens=True))


