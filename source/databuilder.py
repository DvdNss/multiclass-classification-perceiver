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
    parser = parser.parse_args(args)

    # Load tokenizer
    tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")

    for subset in parser.split.split('+'):
        # Load dataset
        if parser.dataset in list_datasets():
            dataset = load_dataset(parser.dataset, split=subset)
        else:
            # Raise error
            raise FileNotFoundError(f"Couldn't find dataset {parser.dataset} or subset {parser.split}. Please use "
                                    f"list_datasets() to list all available datasets. ")

        # Map inputs
        formatted_dataset = dataset.map(lambda row: tokenizer(row['text'], padding="max_length", truncation=True),
                                        batched=True)

        # Build tensor
        formatted_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Save the pt file
        if not os.path.exists(parser.output_dir):
            os.mkdir(parser.output_dir)
        torch.save(formatted_dataset, f"{parser.output_dir}/{subset}.pt".replace("//", "/"))
        print(f"{parser.output_dir}/{subset}.pt file has been saved ({parser.dataset}/{subset}). ")


if __name__ == '__main__':
    main()
