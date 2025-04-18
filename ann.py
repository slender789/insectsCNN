# Loss: 0.4783535897731781, Accuracy: 0.782608687877655

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_images_from_folder(folder, label, image_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('JPG', 'jpeg', 'png')):
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, image_size)
                image = image.astype('float32') / 255.0
                image = image.flatten()
                images.append(image)
                labels.append(label)
    return images, labels

def create_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(12288,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

# Load and preprocess the images
insects_folder = 'all_insects'
no_insects_folder = 'all_noise'

insect_images, insect_labels = load_images_from_folder(insects_folder, label=1)
no_insect_images, no_insect_labels = load_images_from_folder(no_insects_folder, label=0)

# Combine and split the dataset
X = np.array(insect_images + no_insect_images)
y = np.array(insect_labels + no_insect_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the ANN model
model = create_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the model weights
model.save_weights('weights/ann_insect_classifier.weights.h5')
