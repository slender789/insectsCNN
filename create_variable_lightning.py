import cv2
import numpy as np
import os

def adjust_lighting(image_path, brightness_factors, output_folder):
    # Load the original image
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Could not load image {image_path}. Please check the file path.")
        return

    # Get the base name of the original image (with extension)
    base_name = os.path.basename(image_path)

    # Generate and save brighter and darker images
    for factor in brightness_factors:
        # Adjust brightness
        adjusted_image = cv2.convertScaleAbs(original_image, alpha=factor, beta=0)
        
        # Create subfolder for the brightness factor
        factor_folder = os.path.join(output_folder, f"brightness_{factor:.2f}")
        if not os.path.exists(factor_folder):
            os.makedirs(factor_folder)

        # Define output path
        output_path = os.path.join(factor_folder, base_name)
        
        # Save the adjusted image
        cv2.imwrite(output_path, adjusted_image)
        print(f"Saved: {output_path}")

def process_images_in_folder(input_folder, brightness_factors, output_folder):
    # Ensure the input folder exists
    if not os.path.exists(input_folder):
        print("Error: Input folder does not exist.")
        return

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            adjust_lighting(file_path, brightness_factors, output_folder)

if __name__ == "__main__":
    # Path to the folder containing images
    input_folder = "raw_images"  # Replace with your folder path

    # List of brightness factors (greater than 1 for brighter, less than 1 for darker)
    brightness_factors = [0.15, 0.20, 0.30, 0.40, 0.5, 0.75, 1.25, 1.5, 2.0]  # Adjust these values as needed

    # Output folder to save the resultant images
    output_folder = "variable_lighting_images"

    # Process all images in the input folder
    process_images_in_folder(input_folder, brightness_factors, output_folder)
