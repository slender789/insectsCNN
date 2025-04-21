import numpy as np
import cv2
import tensorflow as tf
from resnet_cnn import build_model

def preprocess_image(image_path, img_size=(64, 64), fromPath = False, image=None):
    if fromPath:
        image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def load_model_weights(model, weights_path):
    model.load_weights(weights_path)
    print(f'Model weights loaded from {weights_path}')

def predict_insect(model_weights_path, fromPath, image_path=None, image=None):
    model = build_model()
    
    load_model_weights(model, model_weights_path)
    
    image = preprocess_image(image_path = image_path, fromPath = fromPath, image = image)
    
    if image is None:
        print("Error: Could not read image")
        return None
    
    prediction = model.predict(image)
    
    if prediction[0][0] > 0.5:
        print("Insect detected")
        return True
    else:
        print("No insect detected")
        return False

if __name__ == "__main__":
    test_image_path = 'IMG_5784_40_segment_4.png'
    model_weights_path = 'insect_detection_model.weights.h5'

    predict_insect(model_weights_path, fromPath=True, image_path=test_image_path)
