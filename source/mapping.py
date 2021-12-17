# coding:utf-8
"""
Filename: mapping.py
Author: @DvdNss

Created on 12/17/2021
"""
from typing import List


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
