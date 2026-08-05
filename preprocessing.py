"""Image preprocessing shared by evaluation and simulator inference."""

from pathlib import Path

import cv2
import numpy as np

CROP_TOP = 60
CROP_BOTTOM = 135
MODEL_WIDTH = 200
MODEL_HEIGHT = 66


def preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """Apply the same preprocessing steps used in the training notebook."""
    if image_rgb is None:
        raise ValueError("Image is empty.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB image with 3 channels, got {image_rgb.shape}."
        )
    if image_rgb.shape[0] < CROP_BOTTOM:
        raise ValueError(
            f"Image height must be at least {CROP_BOTTOM}, got {image_rgb.shape[0]}."
        )

    image = image_rgb[CROP_TOP:CROP_BOTTOM, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))
    image = image.astype(np.float32) / 255.0
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image with OpenCV and return it in RGB order."""
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
