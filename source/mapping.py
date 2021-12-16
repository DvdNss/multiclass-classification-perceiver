# coding:utf-8
"""
Filename: map_function.py
Author: @DvdNss

Created on 12/16/2021
"""
from transformers import PerceiverTokenizer

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
          "neutral",
          "optimism",
          "pride",
          "realization",
          "relief",
          "remorse",
          "sadness",
          "surprise"
          ]


def map_function(row: dict) -> dict:
    """
    Map dataset row to given format.

    :param row: dataset row
    :return:
    """

    # Load tokenizer
    tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")

    # Process
    formatted_row = tokenizer(row['text'], padding="max_length", truncation=True)

    return formatted_row


def map_labels(labels: dict):
    label = None
    row = [1, 2]
    keys_to_pop = []
    # Find useless keys
    for key in row:
        if type(row[key]) is not int:
            keys_to_pop.append(key)

    # Pop useless keys
    for key in keys_to_pop:
        row.pop(key)

    # Extract int label
    for idx, key in enumerate(row):
        if row[key] == 1:
            label = idx
