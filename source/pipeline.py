# coding:utf-8
"""
Filename: inference.py
Author: @DvdNss

Created on 12/17/2021
"""
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm


def _map_outputs(predictions):
    """
    Map model outputs to classes.

    :param predictions: model ouptut batch
    :return:
    """

    labels = [
        "admiration",
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
        "neutral"
    ]
    classes = []
    for i, example in enumerate(predictions):
        out_batch = []
        for j, category in enumerate(example):
            out_batch.append(labels[j]) if category > 0.5 else None
        classes.append(out_batch)
    return classes


class MultiLabelPipeline:
    """
    Multi label classification pipeline.
    """

    def __init__(self, model, tokenizer, **kwargs):
        """
        Init MLC pipeline.

        :param model: model to use
        :param tokenizer: tokenizer to use
        :param kwargs: other args
        """

        # Init attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer

    def __call__(self, dataset, batch_size: int = 4):
        """
        Processing pipeline.

        :param dataset: dataset
        :return:
        """

        # Tokenize inputs
        dataset = dataset.map(lambda row: self.tokenizer(row['text'], padding="max_length", truncation=True),
                              batched=True, remove_columns=['text'], desc='Tokenizing')
        dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Define output classes
        classes = []
        mem_logs = []

        with tqdm(dataloader, unit='batches') as progression:
            for batch in progression:
                progression.set_description('Inference')
                # Forward
                outputs = self.model(inputs=batch['input_ids'].to(self.device),
                                     attention_mask=batch['attention_mask'].to(self.device),)

                # Outputs
                predictions = outputs.logits.cpu().detach().numpy()

                # Map predictions to classes
                batch_classes = _map_outputs(predictions)

                for row in batch_classes:
                    classes.append(row)

                # Retrieve memory usage
                memory = round(torch.cuda.memory_reserved(self.device) / 1e9, 2)
                mem_logs.append(memory)

                # Update pbar
                progression.set_postfix(memory=f"{round(sum(mem_logs) / len(mem_logs), 2)}Go")

        return classes
