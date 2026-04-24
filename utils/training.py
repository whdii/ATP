from __future__ import division
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from utils.utils import time_string, print_log
from ATP.utils.activation_tracker import ActivationTracker


def train(data_loader, model, optimizer, epsilon, num_iterations, print_freq=200, use_cuda=True, pretrained_arch=None, surrogate_model_bac=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.generator.train()
    model.surrogate_model.eval()
    surrogate_model_bac.eval()

    end = time.time()
    
    extractor = ActivationTracker(model.surrogate_model, surrogate_model_bac, pretrained_arch.replace("_201", ""))
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()
        count_activate, _, count_activate_l1, _, count_activate_bac, _, count_activate_l1_bac, _ = extractor.get_statis(images, batch_idx)
        
    non_robust_idx, non_robust_idx_l1 = get_non_robust_indices(count_activate, count_activate_l1)
    robust_idx_bac, robust_idx_l1_bac = get_robust_indices(count_activate_bac, count_activate_l1_bac)
    
    data_iterator = iter(data_loader)
    iteration = 0
    
    while iteration < num_iterations:
        input, target = get_next_batch(data_iterator, data_loader)
        data_time.update(time.time() - end)

        if use_cuda:
            input, target = input.cuda(), target.cuda()

        adv_images = model.generator(input)
        loss, output, loss_ff = extractor.loss_function(adv_images, input, target, robust_idx_bac, non_robust_idx, robust_idx_l1_bac, non_robust_idx_l1)
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.generator.uap.data = torch.clamp(model.generator.uap.data, -epsilon, epsilon)
        
        batch_time.update(time.time() - end)
        end = time.time()
        iteration += 1
    
    extractor.clear_hook()


def get_non_robust_indices(count_activate, count_activate_l1):
    num_top = int(len(count_activate) * 0.4)
    num_top_l1 = int(len(count_activate_l1) * 0.4)
    non_robust_idx = np.argsort(count_activate)[:num_top]
    non_robust_idx_l1 = np.argsort(count_activate_l1)[:num_top_l1]
    return non_robust_idx, non_robust_idx_l1


def get_robust_indices(count_activate_bac, count_activate_l1_bac):
    num_top_bac = int(len(count_activate_bac) * 0.4)
    num_top_l1_bac = int(len(count_activate_l1_bac) * 0.4)
    robust_idx_bac = np.argsort(count_activate_bac)[-num_top_bac:]
    robust_idx_l1_bac = np.argsort(count_activate_l1_bac)[-num_top_l1_bac:]
    return robust_idx_bac, robust_idx_l1_bac


def get_next_batch(data_iterator, data_loader):
    try:
        input, target = next(data_iterator)
    except StopIteration:
        data_iterator = iter(data_loader)
        input, target = next(data_iterator)
    return input, target

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
