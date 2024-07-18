# No es la solución final, es para mencoinar que se probó

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size[1]):  # Move vertically
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size[0]):  # Move horizontally
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def count_insects_in_image(image_path, window_size=(100, 100), step_size=(100, 100)):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image")
        return 0
    
    image = image.astype('float32') / 255.0
    insect_count = 0
    
    for (x, y, window) in sliding_window(image, window_size, step_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue
        
        window_expanded = np.expand_dims(window, axis=0)  # Add batch dimension
        prediction = model.predict(window_expanded)
        if prediction[0][0] > 0.5:
            insect_count += 1
            window_filename = f"detected_insect_{insect_count}.jpg"
            window_path = os.path.join(detected_folder, window_filename)
            cv2.imwrite(window_path, window * 255.0)  # Save the window image
            print(f"Saved detected insect to {window_path}")
    
    return insect_count
