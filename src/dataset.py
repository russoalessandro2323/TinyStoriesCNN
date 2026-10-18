from datasets import load_dataset
import numpy as np
import torch

def get_dataset():
    """
        Retrieve the roneneldan/TinyStories dataset
        out: trn_raw_dataset(2119719 rows), val_raw_dataset(21990 rows) <class 'datasets.arrow_dataset.Dataset'>
    """

    trn_raw_dataset = load_dataset("roneneldan/TinyStories", split="train")         
    val_raw_dataset = load_dataset("roneneldan/TinyStories", split="validation")    

    return trn_raw_dataset, val_raw_dataset

