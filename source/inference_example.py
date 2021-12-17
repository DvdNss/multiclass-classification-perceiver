# coding:utf-8
"""
Filename: test.py
Author: @DvdNss

Created on 12/17/2021
"""

from pipeline import MultiLabelPipeline, inputs_to_dataset

model_path = '../model/perceiver-e2-acc0.pt'

# Load pipeline
pipeline = MultiLabelPipeline(model_path=model_path)

# Build a little dataset
inputs = ['This this a test.', 'Another test.', 'The final test.']

# Make inference
print(pipeline(inputs_to_dataset(inputs), batch_size=3))
