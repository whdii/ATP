import os
import numpy as np
from torch.utils.data import Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from .utils import get_labels, concoct_dataset

def get_data_specs(pretrained_dataset):
    dataset_specs = {
        "cifar10": (10, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 32, 3),
        "tiny-imagenet": (201, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 32, 3),
        "tiny-imagenet_201": (201, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 32, 3)
    }
    
    if pretrained_dataset not in dataset_specs:
        raise ValueError("Unsupported dataset")

    return dataset_specs[pretrained_dataset]

def create_transforms(input_size, mean, std):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_transform, test_transform

def get_data(dataset, pretrained_dataset, args):
    num_classes, mean, std, input_size, num_channels = get_data_specs(pretrained_dataset)
    train_transform, test_transform = create_transforms(input_size, mean, std)

    if dataset == 'cifar10':
        train_data = dset.CIFAR10('/root/dataset', train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10('/root/dataset', train=False, transform=test_transform, download=True)

    elif dataset == "tiny-imagenet":
        lab = args.target_class
        ori_train = dset.CIFAR10('/root/tiny-imagenet-200', train=True, transform=train_transform, download=True)
        outter_trainset = os.path.join('/root/tiny-imagenet-200', "train")
        outter_train_data = dset.ImageFolder(root=outter_trainset, transform=train_transform)

        train_labels = get_labels(ori_train)
        train_target_list = np.where(np.array(train_labels) == lab)[0]
        train_target = Subset(ori_train, train_target_list)
        train_data = concoct_dataset(train_target, outter_train_data)
        test_data = train_target

    return train_data, test_data
