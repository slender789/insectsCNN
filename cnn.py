import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, img_size=(100, 100)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
    return image

def load_images_from_folder(folder, label, img_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = preprocess_image(img_path, img_size)
        if image is not None:
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

insect_folder = 'all_insects'
non_insect_folder = 'all_noise'
model_weights_path = 'insect_detection_model.weights.h5'

insect_data, insect_labels = load_images_from_folder(insect_folder, 1)
non_insect_data, non_insect_labels = load_images_from_folder(non_insect_folder, 0)

data = np.vstack((insect_data, non_insect_data))
labels = np.hstack((insect_labels, non_insect_labels))

# data = data.astype('float32') / 255.0

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
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

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

# Save the model weights
model.save_weights(model_weights_path)
print(f'Model weights saved to {model_weights_path}')

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Accuracy: {accuracy * 100:.2f}%')

# Load the model weights for testing (optional)
def load_model_weights(model, weights_path):
    model.load_weights(weights_path)
    print(f'Model weights loaded from {weights_path}')
