import argparse
import json
import logging
import os
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


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
        self.model_max_length = args["model_max_length"]
        self.tokenizer = args["tokenizer"]
        self.force_save = args["force_save"]
        dataset = load_dataset(
            'csv',
            data_files={
                'train': os.path.join(self.data_path, 'train.csv'),
                'val': os.path.join(self.data_path, 'validation.csv'),
                'test': os.path.join(self.data_path, 'test.csv'),
            },
        )
        self.train = dataset['train']
        self.eval = dataset['val']
        self.test = dataset['test']

    def generate_and_tokenize_prompt(self, data_point, tokenizer):
        full_prompt = f"""Given a meaning representation generate a target sentence that utilizes the attributes and 
        attribute values given. The sentence should use all the information provided in the meaning representation. 
        ### Target sentence: {data_point["ref"]}

        ### Meaning representation:
        {data_point["mr"]}
        """
        return tokenizer(full_prompt, truncation=True, max_length=self.model_max_length, padding='max_length')

    def load_and_prepare_and_save_data(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer,
            model_max_length=self.model_max_length,
            padding_side='left',
            add_eos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

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
        "data_path": "",
        "tokenizer": 'mistralai/Mistral-7B-v0.1',
        "model_max_length": 512,
        "force_save": True
    }

    data_preprocessor = DataPreprocessor(*args)

    data_preprocessor.load_and_prepare_and_save_data()


if __name__ == '__main__':
    main()
