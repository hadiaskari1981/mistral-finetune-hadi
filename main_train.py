import os
from mistral_viggo_fine_tune import fine_tune

if __name__ == '__main__':

    project = "mistral-viggo-finetune"

    current_working_dir = os.getcwd()

    saved_dir = f"{current_working_dir}/data"
    os.makedirs(saved_dir, exist_ok=True)

    cache_path = f'{saved_dir}/cache'
    os.makedirs(cache_path, exist_ok=True)

    model_output_path = f"{saved_dir}/{project}"
    os.makedirs(model_output_path, exist_ok=True)

    logs_path = f"{saved_dir}/logs"
    os.makedirs(logs_path, exist_ok=True)

    train_args = {
        "base_model_name": "mistralai/Mistral-7B-v0.3",
        "cache_dir": cache_path,
        "is_inference": False,
        "data_path": "gem/viggo",
        "model_max_length": 512,
        "model_output_dir": model_output_path,
        "logs_dir": logs_path,
        "project_name": project

    }
    finetune_trainer = fine_tune(train_args)

    finetune_trainer.train()

