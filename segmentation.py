import cv2
import numpy as np
import os
import sys
from preprocess import preprocess_image

def segmentation(mask, original_image_path, output_folder=None, min_area=1000, max_area=5000, aspect_ratio_range=(0.2, 5)):
    """
    Perform instance segmentation using connected components, filtering out components that don't meet the criteria.
    Optionally crop and save the filtered components from the original image, ensuring square crops with white padding.

    Parameters:
    - mask: Binary mask image.
    - original_image_path: Path to the original image.
    - output_folder: Directory where cropped images will be saved (if provided).
    - min_area: Minimum area threshold for a component to be considered.
    - max_area: Maximum area threshold for a component to be considered.
    - aspect_ratio_range: Tuple (min_aspect_ratio, max_aspect_ratio) to filter components based on their aspect ratio.

    Returns:
    - output: Image with colored instances.
    - filtered_labels: Label matrix with only the filtered labels.
    - filtered_stats: Statistics for the filtered labels.
    - filtered_centroids: Centroids for the filtered labels.
    - cropped_images: List of NumPy arrays containing square-cropped segments.
    """
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Could not load original image: {original_image_path}")
    
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(original_image_path))[0]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Use a dark background to show the highlighted insects
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # # Use a copy of the original image as the output background
    # output = original_image.copy()
    
    filtered_stats = []
    filtered_centroids = []
    filtered_label_indices = []
    cropped_images = []

    for label in range(1, num_labels):  # Skip background
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        x, y = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
        aspect_ratio = width / height if height != 0 else 0
        saturation = area / (width * height)

        if min_area <= area <= max_area and saturation >= 0.4 and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            output[labels == label] = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            filtered_stats.append(stats[label])
            filtered_centroids.append(centroids[label])
            filtered_label_indices.append(label)

            max_side = max(width, height)
            square_crop = np.ones((max_side, max_side, 3), dtype=np.uint8) * 255
            
            x_offset = (max_side - width) // 2
            y_offset = (max_side - height) // 2
            
            cropped_segment = original_image[y:y+height, x:x+width]
            square_crop[y_offset:y_offset+height, x_offset:x_offset+width] = cropped_segment
            
            cropped_images.append(square_crop)

            if output_folder is not None:
                cropped_filename = os.path.join(output_folder, f"{base_filename}_segment_{label}.png")
                cv2.imwrite(cropped_filename, square_crop)

    filtered_stats = np.array(filtered_stats)
    filtered_centroids = np.array(filtered_centroids)

    filtered_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(filtered_label_indices, start=1):
        filtered_labels[labels == old_label] = new_label

    return output, filtered_labels, filtered_stats, filtered_centroids, cropped_images


if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    for filename in os.listdir(input_folder):
        original_image_path = os.path.join(input_folder, filename)
        
        if not os.path.isfile(original_image_path):
            continue
        
        # Load the image
        image = cv2.imread(original_image_path)
        if image is None:
            sys.exit(1)
        # Preprocess the image
        preprocessed = preprocess_image(image)
#         # Convert the image to grayscale before instance segmentation
#         preprocessed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale here

        # Perform instance segmentation
        instance_mask, labels, stats, centroids = segmentation(preprocessed, original_image_path, output_folder)
        print(f"Saved {len(stats)} cropped segments for image {original_image_path}")
