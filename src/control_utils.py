import sys
import os
import torch
import torchvision.transforms as tt
import numpy as np
from PIL import Image


sys.path.append('../ldm')

import cv2

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def create_image_grid(images: np.ndarray, grid_size=None):
    """
    Create a grid with the fed images
    Args:
        images (np.array): array of images
        grid_size (tuple(int)): size of grid (grid_width, grid_height)
    Returns:
        grid (np.array): image grid of size grid_size
    """
    # Sanity check
    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, c = images.shape
    # If user specifies the grid shape, use it
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
        # If one of the sides is None, then we must infer it (this was divine inspiration)
        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)

    # Otherwise, we can infer it by the number of images (priority is given to grid_w)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    # Sanity check
    assert grid_w * grid_h >= num, 'Number of rows and columns must be greater than the number of images!'
    # Get the grid
    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)
    # Paste each image in the grid
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


def get_edge_hint(
    image,    # source image
    version='cv2',   # diffusion version [1.5, 2.1]
    size=512,
    low_th=50,
    high_th=300,
):
    assert version in ('cv2', 'skimage'), f'<version> has to be cv2 or skimage and not {version}.'

    # if type(image).__name__ != 'PngImageFile':
    #     if image.dtype not in (np.uint8, torch.uint8):
    #         image =  image * 128 + 128

    image = np.array(image).astype(np.uint8)[..., :3]
    min_size = min(image.shape[:2])
    trafo_edges = tt.Compose([tt.ToPILImage(), tt.CenterCrop(min_size), tt.Resize(size)])

    im_edges = np.array(image).astype(np.uint8)
    hint_model = CannyDetector()

    detected_map = hint_model(im_edges, low_threshold=low_th, high_threshold=high_th)
    detected_map = np.array(trafo_edges(detected_map)).astype(np.uint8)
    hint = HWC3(detected_map)
    hint = hint / 255.0

    return hint


def get_canny_edges(image, size=512, low_th=50, high_th=200):
    image = np.array(image).astype(np.uint8)

    low_th = low_th or np.random.randint(50, 100)
    high_th = high_th or np.random.randint(200, 350)
    edges = CannyDetector()(image, low_th, high_th)  # original sized greyscale edges
    edges = edges / 255.
    return edges

