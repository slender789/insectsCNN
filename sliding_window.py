import os
import numpy as np
import cv2
from use_weights import predict_insect


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size[1]):  # Move vertically
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size[0]):  # Move horizontally
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def count_insects_in_image(image_path, store=False, output_dir='counted_insects_window', window_size=(100, 100), step_size=(100, 100)):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image")
        return 0
    
    if store:
        os.makedirs(output_dir, exist_ok=True)
    else:
        cropped_images = []

    image = image.astype('float32') / 255.0
    
    for i, (x, y, window) in enumerate(sliding_window(image, window_size, step_size)):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue
        if store:
            output_path = os.path.join(output_dir, f'circle_{i}.png')
            cv2.imwrite(output_path, window)
        else:
            cropped_images.append(window)
    if not store:
        return cropped_images

if __name__ == "__main__":
    image_path = 'IMG_5790_raw.JPG'  # Replace with the path to your folder containing images
    model_weights_path = 'insect_detection_model.weights.h5'

    insects = 0
    notInsects = 0
    
    candidate_insects_images = count_insects_in_image(image_path)
    print(len(candidate_insects_images))
    for candidate_insect in candidate_insects_images:
        isInsect = predict_insect(model_weights_path, fromPath=False, image=candidate_insect)
        if isInsect:
            insects += 1
        else:
            notInsects += 1
    
    print(f'Insects: {insects}')
    print(f'Not insects: {notInsects}')