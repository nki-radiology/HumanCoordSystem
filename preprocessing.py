"""Preprocessing used by the UCS reference implementation."""

import numpy as np
from skimage.transform import resize


def preprocess_cached_slice(image, image_size=256):
    """Resize and normalize a training slice already shifted to [0, 420]."""
    if image.shape != (image_size, image_size):
        image = resize(image, (image_size, image_size), preserve_range=True)
    return (image / 420.0)[..., None]


def preprocess_hu_volume(volume, image_size=256):
    """Resize, window and normalize a raw-HU volume for inference."""
    processed = []
    for image in volume:
        image = resize(image, (image_size, image_size), preserve_range=True)
        image = (np.clip(image, -120, 300) + 120.0) / 420.0
        processed.append(image[..., None])
    return np.stack(processed).astype(np.float32)


def extract_random_patch(image, patch_size, rng=np.random):
    """Extract a patch and return its top-left (row, column) coordinates."""
    height, width = image.shape[:2]
    row = rng.randint(0, height - patch_size)
    column = rng.randint(0, width - patch_size)
    patch = image[row : row + patch_size, column : column + patch_size]
    return patch, (row, column)

