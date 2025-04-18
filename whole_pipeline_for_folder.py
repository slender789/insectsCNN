import os
import re
import time
from typing import Dict, Tuple

from cnn import predict_insect_image
from segmentation import segmentation
from preprocess import preprocess_image
from whole_pipeline_for_ind import load_image_safe

# Configuration
FOLDER_PATH = 'raw_images'
OUTPUT_FILE = 'docs/detection_results_segmentation.txt'


def extract_number_from_filename(filename: str) -> str | None:
    """Extract the number from a filename matching *_<number>.JPG."""
    match = re.search(r'_(\d+)\.JPG$', filename)
    return match.group(1) if match else None


def query_images_in_folder(folder_path: str) -> Dict[str, str]:
    """Find all JPG files in folder with identifiable numbers."""
    return {
        filename: number
        for filename in os.listdir(folder_path)
        if filename.endswith(".JPG") and (number := extract_number_from_filename(filename))
    }


def process_image_file(image_path: str) -> Tuple[int, int, float, float]:
    """
    Process a single image and return classification results.
    Returns:
    - Number of insects
    - Number of non-insects
    - Time taken to segment
    - Time taken to classify
    """
    start_count_time = time.time()

    image = load_image_safe(image_path)
    preprocessed = preprocess_image(image)

    _, _, _, _, cropped_images = segmentation(
        mask=preprocessed,
        original_image_path=image_path
    )

    segment_time = time.time() - start_count_time

    insects, not_insects = 0, 0
    total_predict_time = 0.0

    for candidate_image in cropped_images:
        start = time.time()
        is_insect = predict_insect_image(candidate_image)
        total_predict_time += time.time() - start

        if is_insect:
            insects += 1
        else:
            not_insects += 1

    return insects, not_insects, segment_time, total_predict_time


def main():
    results = query_images_in_folder(FOLDER_PATH)

    with open(OUTPUT_FILE, 'w') as file:
        for filename, real_insects in results.items():
            image_path = os.path.join(FOLDER_PATH, filename)
            try:
                insects, not_insects, count_time, predict_time = process_image_file(image_path)
            except (FileNotFoundError, ValueError) as e:
                print(f"[ERROR] Skipping '{filename}': {e}")
                continue  # Skip and proceed with others

            file.write(f'Filename: {filename}, Detected Insects: {insects} vs. Real Insects: {real_insects}\n')
            file.write(f'Not insects: {not_insects}\n')
            file.write(f'Total time taken to count insects: {count_time:.2f} seconds\n')
            file.write(f'Total time taken to predict insects: {predict_time:.2f} seconds\n\n')


if __name__ == "__main__":
    main()
