import os
import sys
import time
import cv2

from resnet_cnn import predict_insect_image
from segmentation import segmentation
from preprocess import preprocess_image

# Configuration
IMAGE_PATH = 'close_raw/IMG_5784.JPG'

def load_image_safe(path: str):
    """Safely load image or raise error."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to read image from path: {path}")
    
    return image


def count_insects(cropped_images):
    """Classify cropped images and count insects vs. non-insects."""
    insects = 0
    not_insects = 0
    total_predict_time = 0.0

    for candidate_image in cropped_images:
        start = time.time()
        is_insect = predict_insect_image(candidate_image)
        total_predict_time += time.time() - start

        if is_insect:
            insects += 1
        else:
            not_insects += 1

    return insects, not_insects, total_predict_time


def main():
    start_time = time.time()

    try:
        image = load_image_safe(IMAGE_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    preprocessed = preprocess_image(image)
    _, _, _, _, cropped_images = segmentation(
        mask=preprocessed,
        original_image_path=IMAGE_PATH
    )

    insects, not_insects, prediction_time = count_insects(cropped_images)
    total_time = time.time() - start_time

    print(f"Insects: {insects}")
    print(f"Not insects: {not_insects}")
    print(f"Total time taken to count insects: {total_time:.2f} seconds")
    print(f"Total time taken to predict insects: {prediction_time:.2f} seconds")


if __name__ == "__main__":
    main()
