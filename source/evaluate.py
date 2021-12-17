# coding:utf-8
"""
Filename: evaluate.py
Author: @DvdNss

Created on 12/17/2021
"""
import argparse

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model=None, validation_dataloader=None):
    """
    Evaluate model.
    :return:
    """

    # Init device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and dataloader if called by command line
    if all(item is None for item in [model, validation_dataloader]):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='Model path. ')
        parser.add_argument('--validation_data', help='Path to validaton torch file. ', default='data/validation.pt')
        parser.add_argument('--batch_size', help='Batch size. ', default=4)
        parser = parser.parse_args()

        model = torch.load(parser.model).to(device)
        validation_dataloader = DataLoader(torch.load(parser.validation_data), batch_size=parser.batch_size)

    # Evaluate model
    model.eval()
    accu_logs = []
    with tqdm(validation_dataloader, unit='batches') as progression:
        for batch in progression:
            progression.set_description('Evaluation')
            # Get inputs
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Predict
            outputs = model(inputs=inputs, attention_mask=attention_mask)
            predictions = outputs.logits.cpu().detach().numpy()
            references = batch["targets"].numpy()

            # Binarize predictions
            for i, example in enumerate(predictions):
                for j, category in enumerate(example):
                    predictions[i][j] = 1 if category > 0.5 else 0

            # Compute accuracy for batch and add to buffer
            accuracy = accuracy_score(y_true=references, y_pred=predictions)
            accu_logs.append(accuracy)
            global_acc = sum(accu_logs) / len(accu_logs) * 100
            progression.set_postfix(accuracy=global_acc)

    print("")
    return global_acc


if __name__ == '__main__':
    evaluate()
