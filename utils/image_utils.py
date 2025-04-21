import os
import numpy as np
import cv2
from typing import Callable, Tuple, List, Optional

def default_preprocess(image, image_size=(64, 64), flatten=True):
    """Resize, normalize, and optionally flatten an image."""
    image = cv2.resize(image, image_size)
    image = image.astype('float32') / 255.0
    if flatten:
        image = image.flatten()
    return image

def load_images_from_folder(
    folder: str,
    label: int,
    image_size: Tuple[int, int] = (64, 64),
    flatten: bool = True,
    preprocess_fn: Optional[Callable] = None,
    file_extensions: Tuple[str, ...] = ('JPG', 'jpeg', 'png')
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess images from a folder.

    Parameters:
        folder (str): Path to the folder containing images.
        label (int): Label to assign to all images.
        image_size (tuple): Size to resize images to.
        flatten (bool): Whether to flatten images (for classical ML).
        preprocess_fn (callable): Custom preprocessing function. If None, uses default_preprocess.
        file_extensions (tuple): Allowed file extensions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of images and labels.
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.lower().endswith(file_extensions):
            image = cv2.imread(img_path)
            if image is not None:
                if preprocess_fn is not None:
                    processed = preprocess_fn(image, image_size)
                else:
                    processed = default_preprocess(image, image_size, flatten)
                images.append(processed)
                labels.append(label)
    return np.array(images), np.array(labels)
