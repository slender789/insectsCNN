import numpy as np
import cv2
from utils.image_utils import load_images_from_folder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, img_size=(64, 64)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
    return image

def build_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

if __name__ == "__main__":
    insect_folder = 'all_insects'
    non_insect_folder = 'all_noise'
    model_weights_path = 'weigths/vggnet_cnn.keras'

    # Load dataset
    def preprocess_img_cv2(image, img_size=(64, 64)):
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
        return image

    insect_data, insect_labels = load_images_from_folder(
        insect_folder, 1, image_size=(64, 64), flatten=False, preprocess_fn=preprocess_img_cv2
    )
    non_insect_data, non_insect_labels = load_images_from_folder(
        non_insect_folder, 0, image_size=(64, 64), flatten=False, preprocess_fn=preprocess_img_cv2
    )

    data = np.vstack((insect_data, non_insect_data))
    labels = np.hstack((insect_labels, non_insect_labels))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Build and compile model
    model = build_model()

    lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=50 * len(X_train) // 32)
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(
            model_weights_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # Save the best weights again explicitly (if desired)
    model.save_weights(model_weights_path)
    print(f'Model weights saved to {model_weights_path}')

    # Evaluate
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Predict
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Confusion matrix and report
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    print('Classification Report:')
    print(classification_report(y_test, y_pred))
