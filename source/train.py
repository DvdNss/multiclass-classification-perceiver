# coding:utf-8
"""
Filename: train.py
Author: @DvdNss

Created on 12/17/2021
"""
import argparse
import os

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PerceiverForSequenceClassification, AdamW

from evaluate import evaluate


def main():
    """
    Train language-perceiver given files and arguments.

    :return:
    """

    # TODO: add logger ?? (not even needed, will see)
    # TODO: add wandb
    # TODO: model logs in config file
    # TODO: make labels a script OR global variable ?

    # Create parser and its args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', help='Path to train torch file. ', default='data/train.pt')
    parser.add_argument('--validation_data',
                        help='Path to validaton torch file. If not provided, no validation will occur. ', default=None)
    parser.add_argument('--batch_size', help='Batch size. ', default=1)
    parser.add_argument('--lr', help='Learning rate. ', default=5e-5)
    parser.add_argument('--epochs', help='Number of epochs. ', default=1)
    parser.add_argument('--output_dir', help='Output directory. ', default='model')
    parser = parser.parse_args()

    # Call dataloaders
    train_dataloader = DataLoader(torch.load(parser.train_data), batch_size=parser.batch_size, shuffle=True)
    validation_dataloader = DataLoader(torch.load(parser.validation_data),
                                       batch_size=parser.batch_size) if parser.validation_data is not None else None

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

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PerceiverForSequenceClassification.from_pretrained('deepmind/language-perceiver',
                                                               problem_type='multi_label_classification',
                                                               num_labels=len(labels))

    # Send model to device
    model.to(device)

    # Define optimizer and metric
    optimizer = AdamW(model.parameters(), lr=parser.lr)

    # Train the model
    for epoch in range(int(parser.epochs)):

        # Put model in training mode
        model.train()

        # Init logs
        accu_logs = loss_logs = mem_logs = []

        # Init pbar
        with tqdm(train_dataloader, unit='batches') as progression:

            # Set pbar description
            progression.set_description(f"Epoch {epoch}")

            # Iterate over batches
            for batch in progression:

                # Get inputs
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].type(torch.FloatTensor).to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward, backward & optimizer
                outputs = model(inputs=inputs, attention_mask=attention_mask, labels=targets)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # Evaluate over batch
                if parser.validation_data is not None:

                    # Get predictions and targets
                    predictions = outputs.logits.cpu().detach().numpy()
                    references = batch["targets"].numpy()

                    # Binarize predictions
                    for i, example in enumerate(predictions):
                        for j, category in enumerate(example):
                            predictions[i][j] = 1 if category > 0.5 else 0

                    # Retrieve acc and mem
                    accuracy = accuracy_score(y_true=references, y_pred=predictions)
                    memory = round(torch.cuda.memory_reserved(device) / 1e9, 2)

                    # Append data to logs
                    accu_logs.append(accuracy)
                    loss_logs.append(loss.item())
                    mem_logs.append(memory)

                    # Set logs on pbar
                    progression.set_postfix(loss=round(sum(loss_logs) / len(loss_logs), 3),
                                            accuracy=round(sum(accu_logs) / len(accu_logs) * 100, 1),
                                            memory=f"{round(sum(mem_logs) / len(mem_logs), 2)}Go")
                else:
                    progression.set_postfix(loss=round(sum(loss_logs) / len(loss_logs), 3),
                                            accuracy='disabled',
                                            memory=f"{round(sum(mem_logs) / len(mem_logs), 2)}Go")

        # Create output directory if needed
        if not os.path.exists(parser.output_dir):
            os.mkdir(parser.output_dir)

        # Evaluate and save the model
        if validation_dataloader is not None:
            epoch_acc = evaluate(model=model, validation_dataloader=validation_dataloader)
            torch.save(model, f"{parser.output_dir}/perceiver-e{epoch}-acc{int(epoch_acc)}.pt".replace('//', '/'))
        else:
            torch.save(model, f"{parser.output_dir}/perceiver-e{epoch}.pt".replace('//', '/'))


if __name__ == '__main__':
    main()
