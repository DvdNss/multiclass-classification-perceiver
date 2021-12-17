# coding:utf-8
"""
Filename: dataloader.py
Author: @DvdNss

Created on 12/16/2021
"""

import argparse
import os

import torch
from datasets import load_dataset, list_datasets
from transformers import PerceiverTokenizer

from mapping import map_inputs, map_targets


def main(args=None):
    """
    Load, transform and save a dataset.

    :return:
    """

    # Create parser and its args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name. ')
    parser.add_argument('--split', help="Subsets to use. Separe them by '+' if multiple (ex: train+validation). ",
                        default='train')
    parser.add_argument('--output_dir', help='Output directory. ', default='data')
    parser.add_argument('--max_size', help='Max size of dataset. ', default=None)
    parser = parser.parse_args(args)

    # Load tokenizer
    tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")

    for subset in parser.split.split('+'):
        # Load dataset
        if parser.dataset in list_datasets():
            if parser.max_size is not None:
                dataset = load_dataset(parser.dataset, split=f"{subset}[0:{parser.max_size}]")
            else:
                dataset = load_dataset(parser.dataset, split=subset)
        else:
            # Raise error
            raise FileNotFoundError(f"Couldn't find dataset {parser.dataset} or subset {parser.split}. Please use "
                                    f"list_datasets() to list all available datasets. ")

        # Load labels
        labels = ["admiration",
                  "amusement",
                  "anger",
                  "annoyance",
                  "approval",
                  "caring",
                  "confusion",
                  "curiosity",
                  "desire",
                  "disappointment",
                  "disapproval",
                  "disgust",
                  "embarrassment",
                  "excitement",
                  "fear",
                  "gratitude",
                  "grief",
                  "joy",
                  "love",
                  "nervousness",
                  "optimism",
                  "pride",
                  "realization",
                  "relief",
                  "remorse",
                  "sadness",
                  "surprise",
                  "neutral"]
        id2label = {idx: label for idx, label in enumerate(labels)}
        label2id = {label: idx for idx, label in enumerate(labels)}

        # Map inputs
        formatted_dataset = dataset.map(lambda row: tokenizer(map_inputs(row), padding="max_length", truncation=True),
                                        batched=True, remove_columns=['id', 'text'])

        # Map labels
        formatted_dataset = formatted_dataset.map(lambda row: map_targets(row['labels']), remove_columns=['labels'])

        # Build tensor
        formatted_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'targets'])

        # Save the pt file
        if not os.path.exists(parser.output_dir):
            os.mkdir(parser.output_dir)
        torch.save(formatted_dataset, f"{parser.output_dir}/{subset}.pt".replace("//", "/"))
        print(f"{parser.output_dir}/{subset}.pt file has been saved ({parser.dataset}/{subset}). ")


if __name__ == '__main__':
    main()
