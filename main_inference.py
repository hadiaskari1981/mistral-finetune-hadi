import os
from mistral_viggo_fine_tune import inference

if __name__ == '__main__':
    project = "mistral-viggo-finetune"

    current_working_dir = os.getcwd()

    saved_dir = f"{current_working_dir}/data"
    os.makedirs(saved_dir, exist_ok=True)

    cache_path = f'{saved_dir}/cache'
    os.makedirs(cache_path, exist_ok=True)

    model_output_path = f"{saved_dir}/{project}"
    os.makedirs(model_output_path, exist_ok=True)

    inference_args = {
        "base_model_name": "mistralai/Mistral-7B-v0.3",
        "cache_dir": cache_path,
        "is_inference": True,
        "data_path": "gem/viggo",
        "model_max_length": 512,
        "project_name": project,
        "model_output_path": model_output_path,
        "checkpoint": "checkpoint-1000",
        "target_sentence": "I remember you saying you found Little Big Adventure to be average. Are you not usually "
                           "that into single-player games on PlayStation?"
    }
    inference(inference_args)
