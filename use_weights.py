import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def preprocess_image(image_path, img_size=(100, 100)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def build_model(input_shape=(100, 100, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def load_model_weights(model, weights_path):
    model.load_weights(weights_path)
    print(f'Model weights loaded from {weights_path}')

def predict_insect(image_path, model_weights_path):
    model = build_model()
    
    load_model_weights(model, model_weights_path)
    
    image = preprocess_image(image_path)
    
    if image is None:
        print("Error: Could not read image")
        return None
    
    prediction = model.predict(image)
    
    if prediction[0][0] > 0.5:
        print("Insect detected")
    else:
        print("No insect detected")

test_image_path = 'IMG_5789_test.JPG'
model_weights_path = 'insect_detection_model.weights.h5'

predict_insect(test_image_path, model_weights_path)
