"""
Image and tensor conversion utilities for ComfyUI-Distributed.
"""
import torch
import numpy as np
from PIL import Image

def tensor_to_pil(img_tensor, batch_index=0):
    """Takes a batch of images in tensor form [B, H, W, C] and returns an RGB PIL Image."""
    return Image.fromarray((255 * img_tensor[batch_index].cpu().numpy()).astype(np.uint8))

def pil_to_tensor(image):
    """Takes a PIL image and returns a tensor of shape [1, H, W, C]."""
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If grayscale, add channel dimension
        image = image.unsqueeze(-1)
    return image

def ensure_contiguous(tensor):
    """Ensure tensor is contiguous in memory."""
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor

