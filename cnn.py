import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def preprocess_image_resnet(image_path, img_size=(224, 224)):
    """Read and preprocess an image for ResNet50."""
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, img_size)
        image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


def load_images_from_folder(folder, label, img_size=(224, 224)):
    """Load and preprocess images from a given folder."""
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = preprocess_image_resnet(img_path, img_size)
        if image is not None:
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


def build_model():
    """Build the CNN model with a ResNet50 base."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.7)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def predict_insect_image_path(image_path, model_path="weights/resnet_cnn.keras") -> bool:
    """Classify a single image using the saved model."""
    model = load_model(model_path)
    image = preprocess_image_resnet(image_path)
    if image is None:
        raise ValueError(f"Could not load or preprocess image: {image_path}")
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    return prediction > 0.5

def predict_insect_image(image: np.ndarray, model_path="weights/resnet_cnn.keras") -> bool:
    """
    Classify a single image (already read by cv2) using the saved model.

    Parameters:
    - image: np.ndarray, raw BGR image (as read by cv2).
    - model_path: str, path to the saved model file.

    Returns:
    - bool: True if classified as insect, False otherwise.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input: must be a valid NumPy array.")

    # Resize and preprocess
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Load model and predict
    model = load_model(model_path)
    prediction = model.predict(image)[0][0]
    return prediction > 0.5

def main():
    insect_folder = 'all_insects'
    non_insect_folder = 'all_noise'
    model_weights_path = 'weights/resnet_cnn.keras'

    # Load data
    insect_data, insect_labels = load_images_from_folder(insect_folder, 1)
    non_insect_data, non_insect_labels = load_images_from_folder(non_insect_folder, 0)

    data = np.vstack((insect_data, non_insect_data))
    labels = np.hstack((insect_labels, non_insect_labels))

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))

    # Build and compile model
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    datagen.fit(X_train)

    # Initial training (frozen base)
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=5,
              validation_data=(X_test, y_test))

    # Fine-tuning last layers
    for layer in model.layers[-10:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(model_weights_path, monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train with fine-tuning
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=30,
                        validation_data=(X_test, y_test),
                        class_weight=class_weights,
                        callbacks=[early_stop, lr_scheduler, checkpoint])

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f'Final Accuracy: {accuracy * 100:.2f}%')

    # Prediction analysis
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
