import os
import torch
import numpy as np
import argparse
import multiprocessing
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.nn as nn

from networks.atp import ATP
from utils.data import get_data
from utils.utils import get_model_path
from utils.network import get_network
from utils.training import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a ATP')
    parser.add_argument('--pretrained_dataset', default='tiny-imagenet', choices=[
        'cifar10', 'caltech256', 'tiny-imagenet'],
        help='Used dataset to generate ATP (default: imagenet)')
    parser.add_argument('--target_dataset', default='cifar10',
        help='Used dataset to train the initial model (default: imagenet)')
    parser.add_argument('--pretrained_arch', default='resnet18_201', choices=[
        'resnet18_201', 'DenseNet121_201', 'EfficientNet_201', 'GoogLeNet_201', 'Inceptionv3_201'],
        help='Used model architecture')
    parser.add_argument('--target_arch', default='resnet18', choices=[
        'resnet18', 'DenseNet121', 'EfficientNet', 'GoogLeNet', 'Inceptionv3'])
    parser.add_argument('--pretrained_seed', type=int, default=65,
        help='Seed used in the generation process')
    parser.add_argument('--epsilon', type=float, default=10/255,
        help='Norm restriction of ATP')
    parser.add_argument('--num_iterations', type=int, default=1000,
        help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=256,
        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
        help='Learning Rate (default: 0.001)')
    parser.add_argument('--print_freq', default=20, type=int,
        help='Print frequency (default: 200)')
    parser.add_argument('--ngpu', type=int, default=1,
        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=32,
        help='Number of data loading workers (default: 6)')
    
    args = parser.parse_args()
    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    args.workers = max(1, multiprocessing.cpu_count() - 1)

    return args

def main():
    args = parse_arguments()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    data_train_loader = setup_data_loaders(args)
    
    surrogate_model_adv, surrogate_model_bac, target_network = setup_models(args, device)
    
    warm_up_model(surrogate_model_bac, data_train_loader)
    
    surrogate_model_bac.eval()
    surrogate_model_adv.eval()
    target_network.eval()

    generator, perturbed_net = setup_generator_and_network(args, surrogate_model_adv)
    optimizer = torch.optim.Adam(perturbed_net.parameters(), lr=args.learning_rate)
    
    train(
        data_loader=data_train_loader,
        model=perturbed_net,
        optimizer=optimizer,
        epsilon=args.epsilon,
        num_iterations=args.num_iterations,
        print_freq=args.print_freq,
        use_cuda=args.use_cuda,
        pretrained_arch=args.pretrained_arch,
        pretrained_dataset=args.pretrained_dataset,
        surrogate_model_bac=surrogate_model_bac
    )

def setup_data_loaders(args):
    _, data_train = get_data(args.pretrained_dataset, args.pretrained_dataset, args)
    return DataLoader(
        data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

def setup_models(args, device):
    surrogate_model_adv = get_network(args.pretrained_arch)
    surrogate_model_bac = get_network(args.pretrained_arch)
    target_network = get_network(args.target_arch)

    load_model_weights(surrogate_model_adv, args, device)
    load_model_weights(surrogate_model_bac, args, device)
    load_model_weights(target_network, args, device)

    return surrogate_model_adv, surrogate_model_bac, target_network

def load_model_weights(model, args, device):
    model_path = get_model_path(args.pretrained_dataset, args.pretrained_arch)
    weights_path = os.path.join(model_path, "checkpoint.pth")
    if args.pretrained_dataset != "imagenet":
        network_data = torch.load(weights_path, map_location=device)
        model.load_state_dict(network_data['net'] if args.pretrained_arch != "resnet18_201" else network_data)

def warm_up_model(model, data_loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(params=model.parameters(), lr=0.1)
    for epoch in range(3):
        loss_list = []
        for images, labels in data_loader:
            labels = torch.full(labels.shape, 200, dtype=labels.dtype)
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_list.append(float(loss.data))
            optimizer.step()
        ave_loss = np.average(loss_list)
        print(f'Epoch: {epoch}, Loss: {ave_loss:e}')

def setup_generator_and_network(args, surrogate_model_adv):
    generator = ATP(
        shape=(args.input_size, args.input_size),
        num_channels=args.num_channels,
        mean=args.mean,
        std=args.std,
        use_cuda=args.use_cuda
    )
    perturbed_net = nn.Sequential(OrderedDict([
        ('generator', generator),
        ('surrogate_model', surrogate_model_adv)
    ]))
    return generator, perturbed_net

if __name__ == '__main__':
    main()
