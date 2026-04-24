import numpy as np
from networks.resnet import ResNet18_201, ResNet18

def get_network(model_arch, input_size, num_classes=1000, finetune=False):
    if model_arch == "resnet18_201":
        return ResNet18_201()
    elif model_arch == "resnet18":
        return ResNet18(num_classes=10)
    else:
        raise ValueError("Unsupported model architecture")

def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

def count_parameters(model, trainable_only=True):
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        parameters = model.parameters()
    return sum(np.prod(p.size()) for p in parameters)
