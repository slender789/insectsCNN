import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the image: normalize illumination, threshold, and apply morphological operations.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize illumination
    blurred = cv2.GaussianBlur(gray, (101, 101), 0)
    normalized = cv2.divide(gray, blurred, scale=255)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

def display_preprocessed_image(image_path):
    """
    Load an image, preprocess it, and display the result.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Display the preprocessed image
    cv2.imshow("Preprocessed Image", preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def store_preprocessed_image(image_path, store_image_path = 'preprocessed_image.jpg'):
    """
    Load an image, preprocess it, and display the result.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Store the preprocessed image
    cv2.imwrite(store_image_path, preprocessed)


if __name__ == "__main__":
    # Example usage
    display_preprocessed_image("raw_images/IMG_5820_49.JPG")
