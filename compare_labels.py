import cv2
import os
import numpy as np
from preprocess import preprocess_image
from segmentation import segmentation

def compare_labels(image_folder, ground_truth_folder):
    results = []
    
    # Get list of images in the input folder
    image_files = [f for f in os.listdir(image_folder) if f.startswith("IMG_") and f.endswith(".JPG")]
    
    for image_file in image_files:
        # Extract id_num from the filename
        parts = image_file.split("_")
        if len(parts) < 3:
            continue  # Skip unexpected filenames
        
        id_num = parts[1]
        ground_truth_file = f"IMG_{id_num}_gt.JPG"
        
        # Check if corresponding ground truth image exists
        ground_truth_path = os.path.join(ground_truth_folder, ground_truth_file)
        if not os.path.exists(ground_truth_path):
            print(f"Skipping {image_file}, ground truth not found.")
            continue
        
        # Process the input image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        mask = preprocess_image(image)
        _, _, stats, _ = segmentation(mask, image_path)
        num_labels = len(stats)
        
        # Process the ground truth image
        gt_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Error loading ground truth image: {ground_truth_path}")
            continue
        gt_mask = (gt_mask > 0).astype(np.uint8)  # Ensure it's binary
        _, _, gt_stats, _ = segmentation(gt_mask, ground_truth_path)
        num_gt_labels = len(gt_stats)
        
        # Store comparison result
        results.append({
            "image": image_file,
            "ground_truth": ground_truth_file,
            "detected_labels": num_labels,
            "ground_truth_labels": num_gt_labels,
            "difference": abs(num_labels - num_gt_labels)
        })
    
    return results

if __name__ == "__main__":
    image_folder = "raw_images"  # Update with the actual path
    ground_truth_folder = "ground_truth_highlighted"  # Update with the actual path
    results = compare_labels(image_folder, ground_truth_folder)
    
    # Print results
    for result in results:
        print(result)