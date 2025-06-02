import numpy as np
import torch
import torch.nn.functional as F
import math


def vector_norm(x,keepdim=True):
    dim = tuple(range(1,len(x.shape)))
    return torch.linalg.vector_norm(x, dim=dim,keepdim=keepdim)

def normalize(x, inp_dim,eps=1e-12,norm_detach=False):
    denom = torch.clamp(vector_norm(x,keepdim=True), min=eps)
    if norm_detach:
        denom = denom.detach()
    x = math.sqrt(inp_dim) * x / denom
    return x

def cosine_distance(input, target):
    sim = F.cosine_similarity(input,target)
    dist=(1-sim).mean()
    return dist

def logMSE(input, target):
    loss = F.mse_loss(torch.log1p(input), torch.log1p(target))
    return loss

def normalized_MSE(input, target):
    inp_dim = torch.numel(input[0])
    loss = F.mse_loss(normalize(input,inp_dim), normalize(target,inp_dim))
    return loss

def MSE_normalized_MSE(input, target):
    loss1 = normalized_MSE(input, target)
    loss2 = F.mse_loss(input,target)
    return (loss1+loss2)/2

def normalized_huber(input, target):
    inp_dim = torch.numel(input[0])
    loss = F.smooth_l1_loss(normalize(input,inp_dim), normalize(target,inp_dim))
    return loss

def EMA_smooth(arr, alpha=0.5):
    new_arr = [arr[0]]
    for i in range(1, len(arr)):
        new_data = alpha*new_arr[-1] + (1-alpha)*arr[i]
        new_arr.append(new_data)
    new_arr=np.array(new_arr)
    return new_arr

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    param_size  = param_size / 1024 ** 2
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    buffer_size = buffer_size / 1024 ** 2
    return param_size + buffer_size


class SSIM(torch.nn.Module):
    """SSIM. Modified from:
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window', self._create_window(window_size, self.channel))

    def forward(self, img1, img2):
        assert len(img1.shape) == 4

        channel = img1.size()[1]

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            # window = window.to(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - (window_size // 2)) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=(window_size // 2), groups=channel)
        mu2 = F.conv2d(img2, window, padding=(window_size // 2), groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=(window_size // 2), groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=(window_size // 2), groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=(window_size // 2), groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()

        return ssim_map.mean(1).mean(1).mean(1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return