import torch
import torch.nn as nn

class ATP(nn.Module):
    def __init__(self, shape=(224, 224), num_channels=3, mean=[0., 0., 0.], std=[1., 1., 1.], use_cuda=True):
        super(ATP, self).__init__()

        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.atp = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

        device = torch.device('cuda' if use_cuda else 'cpu')

        self.mean_tensor = torch.tensor(mean).view(1, num_channels, 1, 1).expand(1, num_channels, *shape).to(device)
        self.std_tensor = torch.tensor(std).view(1, num_channels, 1, 1).expand(1, num_channels, *shape).to(device)

    def forward(self, x):
        orig_img = x * self.std_tensor + self.mean_tensor
        adv_orig_img = orig_img + self.atp
        adv_x = (adv_orig_img - self.mean_tensor) / self.std_tensor
        return adv_x
