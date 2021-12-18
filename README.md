<!-- PROJECT LOGO -->
<br />
<p align="center">
<h3 align="center">Multiclass Emotion Classification using DeepMind's Perceiver</h3>
<p align="center">
  <img src="https://github.com/DvdNss/nlp-perceiver/blob/main/resource/home.JPG?raw=true" />
</p>

<!-- ABOUT THE PROJECT -->

## About The Project 

This project aims to set [DeepMind Language Perceiver](https://huggingface.co/deepmind/language-perceiver) usable for most NLP tasks easily.

[__Web App here!__](https://share.streamlit.io/dvdnss/facemaskdetection/main/app.py)


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <ul>
      <li><a href="#structure">Structure</a></li>
      <li><a href="#example">Example</a></li>
    </ul>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->

## Getting Started

### Installation

1. Clone the repo

```shell
git clone https://github.com/DvdNss/nlp-perceiver
```

2. Install requirements

```shell
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->

## Usage

### Structure

* `data/`: contains torch data files
* `source/`: contains main scripts
* `model/`: contains models
* `resource/`: contains readme images
  * `databuilder.py`: loads, transforms and saves datasets
  * `train.py`: training script
  * `mapping.py`: mapping functions
  * `evaluate.py`: evaluation script
  * `pipeline.py`: model pipeline (inference)
  * `inference_example.py`: inference use case
* `app.py`: streamlit app script

### Example

1. Set correct mapping functions in `source/mapping.py` for a given dataset
```python
# Map inputs
def map_inputs(row: dict):
    """
    Map inputs with a given format.

    :param row: dataset row
    :return:
    """

    return row['text']


def map_targets(labels: List[int]):
    """
    Map targets with a given format.

    :param labels: list of labels
    :return:
    """

    targets = [0] * 28
    for label in labels:
        targets[label] = 1

    return {'targets': targets}
```

2. Build the torch files using `source/databuilder.py` script
```shell
python source/databuilder.py --dataset go_emotions --split train+validation --output_dir data --max_size max_size
```
> Once the script stops running, there should be a .pt file in the `output_dir` for each split you selected. 

3. Train your model using `source/train.py` script
```shell
python source/train.py --train_data train_data --validation_data validation_data --batch_size batch_size --lr lr --epochs epochs --output_dir output_dir
```
> A model will be saved in `output_dir` each epoch, which will be named as : \
> `output_dir/perceiver-e<epoch>-acc<eval_acc>.pt`.

4. Evaluate your model using `source/evaluate.py` script
```shell
python source/evaluate.py --model model_path --validation_data validation_data --batch_size batch_size
```

5. Inference using the `source/pipeline.py` script (see use case in `inference_example.py`)
```python
from pipeline import MultiLabelPipeline, inputs_to_dataset

model_path = '../model/perceiver-e2-acc0.pt'

# Load pipeline
pipeline = MultiLabelPipeline(model_path=model_path)

# Build a little dataset
inputs = ['This this a test.', 'Another test.', 'The final test.']

# Make inference
outputs = pipeline(inputs_to_dataset(inputs), batch_size=3)
print(outputs)
```

6. Finally, run streamlit app
```shell
streamlit run app.py
```

<!-- CONTACT -->

## Contact

David NAISSE - [@LinkedIn](https://www.linkedin.com/in/davidnaisse/) - private.david.naisse@gmail.com

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[contributors-url]: https://github.com/Sunwaee/PROJECT_NAME/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[forks-url]: https://github.com/Sunwaee/PROJECT_NAME/network/members

[stars-shield]: https://img.shields.io/github/stars/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[stars-url]: https://github.com/Sunwaee/PROJECT_NAME/stargazers

[issues-shield]: https://img.shields.io/github/issues/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[issues-url]: https://github.com/Sunwaee/PROJECT_NAME/issues

[license-shield]: https://img.shields.io/github/license/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[license-url]: https://github.com/Sunwaee/PROJECT_NAME/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://www.linkedin.com/in/davidnaisse/