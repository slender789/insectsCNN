import cv2
import os
import numpy as np
from preprocess import preprocess_image

def calculate_iou(mask_gt, mask_pred):
    """
    Calculate Intersection over Union (IoU) between ground truth and predicted masks.
    """
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def calculate_precision_recall_accuracy(mask_gt, mask_pred):
    """
    Calculate precision, recall, and accuracy between ground truth and predicted masks.
    """
    true_positives = np.sum(np.logical_and(mask_gt == 1, mask_pred == 1))
    false_positives = np.sum(np.logical_and(mask_gt == 0, mask_pred == 1))
    false_negatives = np.sum(np.logical_and(mask_gt == 1, mask_pred == 0))
    true_negatives = np.sum(np.logical_and(mask_gt == 0, mask_pred == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)

    return precision, recall, accuracy

def resize_masks_to_same_size(gt_masks, pred_masks):
    """
    Resize all ground truth and predicted masks to the same size.
    """
    min_height = min(img.shape[0] for img in gt_masks + pred_masks)
    min_width = min(img.shape[1] for img in gt_masks + pred_masks)
    resized_gt_masks = [cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_NEAREST) for img in gt_masks]
    resized_pred_masks = [cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_NEAREST) for img in pred_masks]
    return resized_gt_masks, resized_pred_masks

def evaluate_model(ground_truth_masks, predicted_masks):
    """
    Evaluate the model using IoU, precision, recall, and accuracy.
    """
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    num_images = len(ground_truth_masks)

    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        iou = calculate_iou(gt_mask, pred_mask)
        precision, recall, accuracy = calculate_precision_recall_accuracy(gt_mask, pred_mask)

        total_iou += iou
        total_precision += precision
        total_recall += recall
        total_accuracy += accuracy

    avg_iou = total_iou / num_images
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_accuracy = total_accuracy / num_images

    return avg_iou, avg_precision, avg_recall, avg_accuracy

def process_and_evaluate(image_folder, ground_truth_folder):
    ground_truth_masks = []
    predicted_masks = []

    image_files = [f for f in os.listdir(image_folder) if f.startswith("IMG_") and f.endswith(".JPG")]
    
    for image_file in image_files:
        parts = image_file.split("_")
        if len(parts) < 3:
            continue
        id_num = parts[1]
        ground_truth_file = f"IMG_{id_num}_gt.JPG"

        ground_truth_path = os.path.join(ground_truth_folder, ground_truth_file)
        image_path = os.path.join(image_folder, image_file)
        
        if not os.path.exists(image_path) or not os.path.exists(ground_truth_path):
            print(f"either {image_path} or {ground_truth_path} don't exist")
            continue
        
        image = cv2.imread(image_path)
        image = preprocess_image(image)
        gt_image = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

        if image is None or gt_image is None:
            continue

        ground_truth_masks.append(gt_image)
        predicted_masks.append(image)

    # Resize masks to the same size
    ground_truth_masks, predicted_masks = resize_masks_to_same_size(ground_truth_masks, predicted_masks)

    # Ensure masks are binary (0 or 1)
    ground_truth_masks = [(mask > 0).astype(np.uint8) for mask in ground_truth_masks]
    predicted_masks = [(mask > 0).astype(np.uint8) for mask in predicted_masks]

    # Evaluate the model
    avg_iou, avg_precision, avg_recall, avg_accuracy = evaluate_model(ground_truth_masks, predicted_masks)

    # Print results
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print("=======================================")

if __name__ == "__main__":
    # image_folder = "raw_images"
    image_folders = [
        "variable_lighting_images/brightness_0.15",
        "variable_lighting_images/brightness_0.20",
        "variable_lighting_images/brightness_0.30",
        "variable_lighting_images/brightness_0.40",
        "variable_lighting_images/brightness_0.50",
        "variable_lighting_images/brightness_0.75",
        "variable_lighting_images/brightness_1.25",
        "variable_lighting_images/brightness_1.50",
    ]
    for image_folder in image_folders:
        print(f"The folder to test the brightness variation: {image_folder}")
        ground_truth_folder = "ground_truth_highlighted"
        process_and_evaluate(image_folder, ground_truth_folder)
