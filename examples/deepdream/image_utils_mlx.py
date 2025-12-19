# image_utils_mlx.py
import mlx.core as mx
import numpy as np
import cv2 as cv
import os

IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])


def read_image_mlx(img_path, target_shape=None):
    """Read image and convert to MLX tensor"""
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')

    img = cv.imread(img_path)[:, :, ::-1]  # BGR to RGB

    # Resize if needed
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    img = (img - IMAGENET_MEAN.tolist()) / IMAGENET_STD.tolist()

    # Convert to MLX: (H, W, C) -> add batch dim -> (1, H, W, C)
    img_tensor = mx.array(img)[None, ...]

    return img_tensor


def write_image_mlx(img_path, img_tensor):
    """Write MLX tensor to image file"""
    # Remove batch dimension and convert to numpy
    img = np.array(img_tensor[0])  # (H, W, C)

    # Denormalize
    mean = IMAGENET_MEAN.tolist()
    std = IMAGENET_STD.tolist()
    img = (img * std) + mean

    # Clip and convert to uint8
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    cv.imwrite(img_path, img[:, :, ::-1])