import cv2
import numpy as np
import os

# Define the noise filters
def apply_poisson_noise(image, scale_factor=1.0):
    """Apply Poisson noise."""
    noisy = np.random.poisson(image.astype(np.float32) * scale_factor) / scale_factor
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def apply_salt_and_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    """Apply Salt and Pepper noise."""
    noisy = image.copy()
    total_pixels = image.size
    num_salt = int(total_pixels * amount * salt_vs_pepper)
    num_pepper = int(total_pixels * amount * (1.0 - salt_vs_pepper))

    # Add Salt (white) noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    # Add Pepper (black) noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    return noisy

def apply_gaussian_noise(image, mean=0, std=25):
    """Apply Gaussian noise."""
    gaussian = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy_image

def apply_speckle_noise(image, std=0.1):
    """Apply Speckle noise."""
    noise = np.random.randn(*image.shape) * std
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Define the noise parameters
poisson_params = [1.0, 5.0, 10.0, 15.0, 20.0]  # Scale factors for Poisson noise
salt_pepper_params = [(0.02, 0.5), (0.05, 0.3), (0.1, 0.5), (0.02, 0.7), (0.1, 0.8)]  # (amount, salt_vs_pepper)
gaussian_params = [(0, 25), (10, 30), (20, 35), (5, 50), (0, 60)]  # (mean, std)
speckle_params = [0.05, 0.1, 0.2, 0.3, 0.4]  # Standard deviations for Speckle noise

# Process the image
def process_and_save_images(image_path, output_folder):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Image at {image_path} not found!")
        return

    # Convert to RGB for display in matplotlib (if needed)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get base name and directory to save the results
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    # Apply Poisson noise with different scale factors and save images
    for scale_factor in poisson_params:
        noisy_image = apply_poisson_noise(image_rgb, scale_factor)
        new_name = f"{name}_poisson_{scale_factor}{ext}"
        cv2.imwrite(os.path.join(output_folder, new_name), cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))

    # Apply Salt & Pepper noise with different parameter combinations and save images
    for amount, salt_vs_pepper in salt_pepper_params:
        noisy_image = apply_salt_and_pepper_noise(image_rgb, amount, salt_vs_pepper)
        new_name = f"{name}_salt_pepper_{amount}_{salt_vs_pepper}{ext}"
        cv2.imwrite(os.path.join(output_folder, new_name), cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))

    # Apply Gaussian noise with different mean and std combinations and save images
    for mean, std in gaussian_params:
        noisy_image = apply_gaussian_noise(image_rgb, mean, std)
        new_name = f"{name}_gaussian_{mean}_{std}{ext}"
        cv2.imwrite(os.path.join(output_folder, new_name), cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))

    # Apply Speckle noise with different standard deviations and save images
    for std in speckle_params:
        noisy_image = apply_speckle_noise(image_rgb, std)
        new_name = f"{name}_speckle_{std}{ext}"
        cv2.imwrite(os.path.join(output_folder, new_name), cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))

    print(f"Processed and saved images for {base_name}.")

# Example Usage
def batch_process_images(input_folder, output_folder):
    # Check if output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            process_and_save_images(image_path, output_folder)

# Define the folders
input_folder = "all_noise"  # Update to the folder where your images are
output_folder = "all_noise"  # Update to the folder to save resultant images

# Run the batch processing
batch_process_images(input_folder, output_folder)
