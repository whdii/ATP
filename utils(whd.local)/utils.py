import os
import torch
from torch.utils.data import Dataset
import numpy as np

RESULT_PATH = '/root/ATP/result'
MODEL_PATH = '/root/ATP/model'
PROJECT_PATH = '/root/ATP/'

def get_model_path(dataset_name, network_arch):
    model_path = os.path.join(MODEL_PATH, network_arch, dataset_name)
    os.makedirs(model_path, exist_ok=True)
    return model_path

class LabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)
    
class CombinedDataset(Dataset):
    def __init__(self, target_dataset, outer_dataset):
        self.target_dataset = target_dataset
        self.outer_dataset = outer_dataset

    def __getitem__(self, idx):
        if idx < len(self.outer_dataset):
            img, label = self.outer_dataset[idx]
        else:
            img = self.target_dataset[idx - len(self.outer_dataset)][0]
            label = len(self.outer_dataset.classes)
        return img, label

    def __len__(self):
        return len(self.target_dataset) + len(self.outer_dataset)
