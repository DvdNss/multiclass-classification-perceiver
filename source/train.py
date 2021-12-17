# coding:utf-8
"""
Filename: train.py
Author: @DvdNss

Created on 12/17/2021
"""
import argparse

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PerceiverForSequenceClassification, AdamW


def main(args=None):
    """
    Train language-perceiver given files and arguments.

    :param args:
    :return:
    """

    # TODO: add logger
    # TODO: add wandb
    # Create parser and its args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', help='Path to train torch file. ', default='data/train.pt')
    parser.add_argument('--validation_data',
                        help='Path to validaton torch file. If not provided, no validation will occur. ', default=None)
    parser.add_argument('--batch_size', help='Batch size. ', default=1)
    parser.add_argument('--lr', help='Learning rate. ', default=5e-5)
    parser.add_argument('--epochs', help='Number of epochs. ', default=1)
    parser = parser.parse_args()

    # Call dataloaders
    train_dataset = torch.load(parser.train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=parser.batch_size, shuffle=True)
    if parser.validation_data is not None:
        validation_dataset = torch.load(parser.validation_data)
        train_dataloader = DataLoader(validation_dataset, batch_size=parser.batch_size)

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

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PerceiverForSequenceClassification.from_pretrained('deepmind/language-perceiver')

    # Send model to device
    model.to(device)

    # Train the model
    optimizer = AdamW(model.parameters(), lr=parser.lr)
    model.train()
    for epoch in range(parser.epochs):
        print("Epoch: ", epoch)
        for batch in tqdm(train_dataloader):
            # Get inputs
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            print(inputs, targets)
            # Zero gradients
            optimizer.zero_grad()

            # Fwd + bwd + opt
            outputs = model(inputs=inputs, attention_mask=attention_mask, labels=targets)
            loss = torch.nn.BCEWithLogitsLoss()(outputs[0], targets)
            loss.backward()
            optimizer.step()

            # Evaluate if validation_data specified
            if parser.validation_data is not None:
                predictions = outputs.logits.cpu().detach().numpy()
                accuracy = accuracy_score(y_true=batch["labels"].numpy(), y_pred=predictions)
                print(f"Loss: {loss.item()}, Accuracy: {accuracy}")
            else:
                print(f"Loss: {loss.item()}")
                print("Skipping validation since no validation data has been provided... ")


if __name__ == '__main__':
    main()
