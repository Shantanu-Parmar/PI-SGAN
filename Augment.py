import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
from scipy.ndimage import gaussian_filter
from torchvision.transforms import functional as F
import logging
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Input and output directories
input_dir = "unaug_ground_pngs"
output_dir = "augmented_ground_truths"
os.makedirs(output_dir, exist_ok=True)

# Astronomical parameters (adjust based on your dataset)
BASE_BRIGHTNESS_MEAN = 0.9  # Base brightness level (0-1)
BRIGHTNESS_NOISE_STD = 0.1  # Gaussian noise std for environmental effects
TURBULENCE_SIGMA = 1.5  # Gaussian blur sigma for atmospheric turbulence
SKY_GLOW_INTENSITY = 0.05  # Background noise intensity

def augment_image(image_path, obj_name):
    # Load image with high quality
    original = Image.open(image_path).convert("RGB")
    original_width, original_height = original.size

    # 1. Rotation and Flipping (6 variations)
    rotations = [0, 90, 180, 270]
    flips = [False, True]  # Horizontal flip
    base_images = []
    for rot in rotations:
        for flip_h in flips:
            img = original.rotate(rot, expand=True, resample=Image.BICUBIC)
            if flip_h:
                img = ImageOps.flip(img)
            base_images.append(img)

    # 2. Brightness Variation (3 levels)
    brightness_variations = []
    for base_img in base_images:
        for _ in range(3):  # 3 brightness levels
            enhancer = ImageEnhance.Brightness(base_img)
            # Gaussian noise for astronomical brightness variation
            noise_factor = np.random.normal(BASE_BRIGHTNESS_MEAN, BRIGHTNESS_NOISE_STD)
            bright_img = enhancer.enhance(max(0, min(1, noise_factor)))
            brightness_variations.append(bright_img)

    # 3. Scaling (2 levels: zoom in, zoom out)
    scaled_images = []
    for bright_img in brightness_variations:
        for scale_factor in [0.8, 1.2]:  # Zoom out, zoom in
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            scaled_img = bright_img.resize((new_width, new_height), Image.BICUBIC)
            scaled_images.append(scaled_img)

    # 4. Astronomical Augmentations
    augmented_images = []
    for scaled_img in scaled_images:
        # Atmospheric turbulence (Gaussian blur)
        img_array = np.array(scaled_img)
        blurred = gaussian_filter(img_array, sigma=TURBULENCE_SIGMA)
        blurred_img = Image.fromarray(blurred.astype(np.uint8))

        # Sky glow (add low-intensity noise)
        noise = np.random.normal(0, SKY_GLOW_INTENSITY, img_array.shape)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img_array)

        augmented_images.append(noisy_img)

    # Save all 36 images
    for i, aug_img in enumerate(augmented_images):
        rot_idx = i // (3 * 2)  # Rotation/flip index
        bright_idx = (i // 2) % 3  # Brightness index
        scale_idx = i % 2  # Scale index
        suffix = f"_rot{rotations[rot_idx % len(rotations)]}"
        suffix += "_flip" if flips[rot_idx // len(rotations)] else ""
        suffix += f"_bright{bright_idx}"
        suffix += "_scale0.8" if scale_idx == 0 else "_scale1.2"
        suffix += "_aug"
        save_path = os.path.join(output_dir, obj_name, f"{os.path.basename(image_path).split('.')[0]}{suffix}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        aug_img.save(save_path, "PNG", quality=95)

def process_dataset():
    for obj in os.listdir(input_dir):
        obj_dir = os.path.join(input_dir, obj)
        if os.path.isdir(obj_dir):
            for img_file in os.listdir(obj_dir):
                if img_file.lower().endswith(".png"):
                    image_path = os.path.join(obj_dir, img_file)
                    logger.info(f"Augmenting {img_file} for object {obj}")
                    augment_image(image_path, obj)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("augmentation.log"), logging.StreamHandler()])
    logger = logging.getLogger()
    process_dataset()
    logger.info("Augmentation completed!")