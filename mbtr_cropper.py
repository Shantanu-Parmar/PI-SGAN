import os
import shutil
import logging
from PIL import Image
import csv
import numpy as np
from pathlib import Path

# Configuration
BASE_DIR = "MobilTelesco_Processed"
ORIGINAL_IMAGES_DIR = r"D:/Shantanu/MBTR/sky-survey/Images"  # Adjusted path with raw string
LABELS_DIR = r"D:/Shantanu/MBTR/sky-survey/Labels"  # Adjusted path with raw string
CROP_SIZE = 800
CLASS_NAMES = ["Pleiades", "Jupiter", "Betelgeuse", "Aldebaran", "Zeta_Tauri", "Elnath", "Hassaleh", "Bellatrix"]

# Set up logging and ensure directories exist
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "metadata"), exist_ok=True)  # Added explicit creation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "preprocessing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Create directory structure
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BASE_DIR, "crops", split), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "original_images", split), exist_ok=True)
    for obj in CLASS_NAMES:
        os.makedirs(os.path.join(BASE_DIR, "crops", split, obj), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "labels", split, obj), exist_ok=True)

# Copy original images as backup
for split in ["train", "val", "test"]:
    src_dir = os.path.join(ORIGINAL_IMAGES_DIR, split)
    dst_dir = os.path.join(BASE_DIR, "original_images", split)
    for img_file in os.listdir(src_dir):
        if img_file.endswith((".jpg", ".jpeg", ".png")):
            shutil.copy(os.path.join(src_dir, img_file), os.path.join(dst_dir, img_file))

# Preprocessing function
def preprocess_crops():
    crop_mapping = []
    for split in ["train", "val", "test"]:
        label_dir = os.path.join(LABELS_DIR, split)
        image_dir = os.path.join(ORIGINAL_IMAGES_DIR, split)
        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue
            img_name = os.path.splitext(label_file)[0]
            img_path = os.path.join(image_dir, f"{img_name}.jpg")  # Adjust extension if needed
            if not os.path.exists(img_path):
                logger.warning(f"Image {img_path} not found, skipping {label_file}")
                continue

            # Read image and get size with enhanced error handling
            img = None
            try:
                with Image.open(img_path) as temp_img:
                    img = temp_img.copy()  # Create a copy to avoid file handle issues
                    img_width, img_height = img.size
                    logger.info(f"Processing {img_path}, size: {img_width}x{img_height}")
            except (IOError, ValueError, AttributeError) as e:
                logger.error(f"Failed to open or process {img_path}: {e}, skipping")
                continue
            if img is None:
                logger.error(f"Image {img_path} could not be loaded, skipping")
                continue

            # Read labels
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            crop_idx = 1
            for line in lines:
                try:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    class_id = int(class_id)
                    if 0 <= class_id < len(CLASS_NAMES):
                        obj_name = CLASS_NAMES[class_id]
                    else:
                        logger.warning(f"Invalid class_id {class_id} in {label_file}, skipping line")
                        continue

                    # Convert normalized coordinates to pixels
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    half_size = CROP_SIZE // 2

                    # Define crop bounds, clip to image edges
                    left = max(0, int(x_center_px - half_size))
                    top = max(0, int(y_center_px - half_size))
                    right = min(img_width, int(x_center_px + half_size))
                    bottom = min(img_height, int(y_center_px + half_size))

                    # Ensure 200x200, pad if necessary
                    crop_width = right - left
                    crop_height = bottom - top
                    if crop_width < CROP_SIZE or crop_height < CROP_SIZE:
                        logger.warning(f"Crop for {img_name} {obj_name} too small, padding with black")
                        cropped_img = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (0, 0, 0))
                        img_box = (max(0, half_size - int(x_center_px)), max(0, half_size - int(y_center_px)),
                                 min(CROP_SIZE, half_size + (img_width - int(x_center_px))),
                                 min(CROP_SIZE, half_size + (img_height - int(y_center_px))))
                        cropped_img.paste(img.crop((left, top, right, bottom)), img_box)
                    else:
                        cropped_img = img.crop((left, top, right, bottom))

                    # Resize to exactly 200x200 if clipped
                    if crop_width != CROP_SIZE or crop_height != CROP_SIZE:
                        cropped_img = cropped_img.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)

                    # Save crop
                    crop_filename = f"{img_name}_{obj_name}_{crop_idx}.jpg"
                    crop_path = os.path.join(BASE_DIR, "crops", split, obj_name, crop_filename)
                    cropped_img.save(crop_path)
                    logger.info(f"Saved crop to {crop_path}")

                    # Update label for crop (center at ~100, 100 in 200x200)
                    new_x_center = 0.5  # Center of 200x200
                    new_y_center = 0.5
                    new_width = width * (img_width / CROP_SIZE)  # Scale normalized width
                    new_height = height * (img_height / CROP_SIZE)
                    new_label = f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}/n"
                    label_crop_path = os.path.join(BASE_DIR, "labels", split, obj_name, crop_filename.replace(".jpg", ".txt"))
                    with open(label_crop_path, 'w') as f:
                        f.write(new_label)

                    # Log mapping
                    crop_mapping.append([img_path, crop_path, x_center_px, y_center_px])

                    crop_idx += 1
                except ValueError as e:
                    logger.error(f"Error parsing line in {label_file}: {line}, {e}")
                    continue

    # Save crop mapping
    with open(os.path.join(BASE_DIR, "metadata", "crop_mapping.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["original_image", "crop_image", "x_center_px", "y_center_px"])
        writer.writerows(crop_mapping)

    logger.info("Preprocessing completed")

if __name__ == "__main__":
    preprocess_crops()
