import csv
import cv2
import os
import sys
import time
from segmentation import segmentation
from preprocess import preprocess_image
from whole_pipeline_for_ind import count_insects

def process_images(input_folder, results_file="docs/whole_pipeline_results.csv"):
    results = []

    # Check if results file exists, create it if it doesn't
    if not os.path.exists(results_file):
        with open(results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'y_truth', 'y_pred', 'elapsed_time'])  # header

    for filename in os.listdir(input_folder):
        original_image_path = os.path.join(input_folder, filename)

        if not os.path.isfile(original_image_path):
            continue

        # Extract the ground truth insect count from the filename
        try:
            y_truth = int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            print(f"Error extracting ground truth from {filename}, skipping.")
            continue

        begin = time.time()

        # Load and preprocess the image
        image = cv2.imread(original_image_path)
        if image is None:
            sys.exit(1)
        preprocessed = preprocess_image(image)

        # Perform instance segmentation
        _, _, _, _, cropped_images = segmentation(preprocessed, original_image_path)

        insects, _, _ = count_insects(cropped_images)

        end = time.time() - begin

        # Count the number of segments (predicted insect count)
        y_pred = insects

        # Store the result (filename, y_truth, y_pred, elapsed_time)
        results.append([filename, y_truth, y_pred, end])

        # Print feedback
        print(f"Processed {filename}: Ground truth = {y_truth}, Predicted = {y_pred}, Elapsed = {end:.2f}s")

    # Save results to CSV file
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    input_folder = "raw_images"
    process_images(input_folder)
