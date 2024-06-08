import argparse
import json
import logging
import os
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import sentencepiece


def save_dataset(dataset, data_path, tag='train'):

    project_name = 'mistral'

    metadata = {
        'dataset-versions': [
            {
                'uri': f'dataset://viggo/{project_name}_{tag}',
                'targeting_aliases': [f'dev_{tag}'],
                'tags': ['dev', 'mistral'],
            },
        ],
    }
    out = f'encoded_{tag}'
    save_path = os.path.join(data_path, out)
    dataset.save_to_disk(save_path)

    for file in os.listdir(save_path):
        with open(os.path.join(data_path, f'{file}.metadata.json'), 'w') as outfile:
            json.dump(metadata, outfile)


class DataPreprocessor:
    def __init__(self, args):
        self.data_path = args["data_path"]
        print(self.data_path)
        self.model_max_length = args["model_max_length"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            args["tokenizer"],
            model_max_length=self.model_max_length,
            padding_side='left',
            add_eos_token=True,
            token="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"
        )

        self.force_save = args["force_save"]

        self.train = load_dataset(self.data_path, split='train', trust_remote_code=True)
        self.eval = load_dataset(self.data_path, split='validation', trust_remote_code=True)
        self.test = load_dataset(self.data_path, split='test', trust_remote_code=True)

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
        {data_point["target"]}

        ### Meaning representation:
        {data_point["meaning_representation"]}
        """
        return self.tokenizer(full_prompt, truncation=True, max_length=self.model_max_length, padding='max_length')

    def load_and_prepare_and_save_data(self):

        self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenized_train = self.train.map(self.generate_and_tokenize_prompt)
        tokenized_val = self.eval.map(self.generate_and_tokenize_prompt)
        cwd = os.getcwd()
        if self.force_save:
            save_dataset(tokenized_train, cwd, 'train')
            save_dataset(tokenized_val,  cwd, 'val')
            save_dataset(self.test, cwd, 'test')
        else:
            return tokenized_train, tokenized_val, self.test


def main():
    logging.basicConfig(level=logging.INFO)

    # Add arguments based on your script's needs
    args = {
        "data_path": "gem/viggo",
        "tokenizer": "mistralai/Mistral-7B-v0.3",
        "model_max_length": 512,
        "force_save": True
    }

    data_preprocessor = DataPreprocessor(args)

    data_preprocessor.load_and_prepare_and_save_data()


if __name__ == '__main__':
    main()
