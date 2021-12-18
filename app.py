# coding:utf-8
"""
Filename: app.py
Author: @DvdNss

Created on 12/18/2021
"""
import os

import gdown as gdown
import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize

from source.pipeline import MultiLabelPipeline, inputs_to_dataset


def download_models(ids):
    """
    Download all models.

    :param ids: name and links of models
    :return:
    """

    # Download sentence tokenizer
    nltk.download('punkt')

    # Download model from drive if not stored locally
    with st.spinner('Downloading models, this may take a minute...'):
        for key in ids:
            if not os.path.isfile(f"model/{key}.pt"):
                url = f"https://drive.google.com/uc?id={ids[key]}"
                gdown.download(url=url, output=f"model/{key}.pt")


@st.cache
def load_labels():
    """
    Load model labels.

    :return:
    """

    return [
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


@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """
    Load model and cache it.

    :param model_path: path to model
    :return:
    """

    model = MultiLabelPipeline(model_path=model_path)

    return model


# Page config
st.set_page_config(layout="centered")
st.title("Multiclass Emotion Classification")
st.write("DeepMind Language Perceiver for Multiclass Emotion Classification (Eng). ")

# Variables
ids = {'perceiver-go-emotions': '15m-p0Pwwnh3STi7zXHkKr9HFxliGJikU'}
labels = load_labels()

# Download all models from drive
download_models(ids)

# Display labels
st.markdown(f"__Labels:__ {', '.join(labels)}")

# Model selection
left, right = st.columns([4, 2])
inputs = left.text_area('', max_chars=2048, placeholder='Write something here to see what happens! ')
model_path = right.selectbox('', options=[k for k in ids], index=0, help='Model to use. ')
split = right.checkbox('Split into sentences')
model = load_model(model_path=f"model/{model_path}.pt")
right.write(model.device)

if split:
    if not inputs.isspace():
        with st.spinner('Processing text... This may take a while.'):
            left.write(model(inputs_to_dataset(sent_tokenize(inputs)), batch_size=1))
else:
    if not inputs.isspace():
        with st.spinner('Processing text... This may take a while.'):
            left.write(model(inputs_to_dataset([inputs]), batch_size=1))
