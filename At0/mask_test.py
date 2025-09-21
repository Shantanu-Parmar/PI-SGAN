import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg
import os
from sklearn.model_selection import train_test_split
import shutil

# Base directory (adjust to your local dataset root)
base_dir = r'D:/Shantanu/MBTR/DATA/Labelled/8-Classes'
output_base_dir = r'D:/Shantanu/MBTR/DATA'  # Base directory for outputs
mask_dir = os.path.join(output_base_dir, 'Masks')
image_dir = os.path.join(output_base_dir, 'Images')
label_dir = os.path.join(output_base_dir, 'Labels')

# Create output directories if they don't exist
for dir_path in [mask_dir, image_dir, label_dir]:
    os.makedirs(dir_path, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(mask_dir, split), exist_ok=True)
    os.makedirs(os.path.join(image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_dir, split), exist_ok=True)

# Load annotations: List of [class_id, xc, yc, w, h]
def load_annotations(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

# Function to compute mask for a given bbox
def create_mask(img, annot, pad=0.4, sigma=1.5, thresh_k=1.0, erosion_iter=1, dilation_iter=3):
    height, width = img.shape[:2]
    class_id, xc, yc, w, h = annot
    
    x_min = int((xc - w/2) * width)
    y_min = int((yc - h/2) * height)
    x_max = int((xc + w/2) * width)
    y_max = int((yc + h/2) * height)
    
    x_min_p = max(0, x_min - int(w * width * pad))
    y_min_p = max(0, y_min - int(h * height * pad))
    x_max_p = min(width, x_max + int(w * width * pad))
    y_max_p = min(height, y_max + int(h * height * pad))
    
    crop = img[y_min_p:y_max_p, x_min_p:x_max_p]
    gray = np.mean(crop, axis=2)
    
    blurred = ndimage.gaussian_filter(gray, sigma=sigma)
    thresh = np.mean(blurred) + thresh_k * np.std(blurred)
    binary = blurred > thresh
    
    binary = ndimage.binary_erosion(binary, iterations=erosion_iter)
    binary = ndimage.binary_dilation(binary, iterations=dilation_iter)
    
    labels, num = ndimage.label(binary)
    if num > 0:
        sizes = ndimage.sum(binary, labels, range(1, num+1))
        max_label = np.argmax(sizes) + 1
        mask = (labels == max_label).astype(np.uint8)
    else:
        mask = np.zeros_like(binary, dtype=np.uint8)
    
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y_min_p:y_max_p, x_min_p:x_max_p] = mask
    
    return full_mask, (x_min, y_min, x_max, y_max), class_id

# Find all image-annotation pairs
image_files = []
annot_files = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(root, file)
            annot_path = os.path.join(root, file.replace('.jpg', '.txt'))
            if os.path.exists(annot_path):
                image_files.append(img_path)
                annot_files.append(annot_path)

# Split dataset (80:15:5)
train_images, temp_images, train_annot, temp_annot = train_test_split(image_files, annot_files, test_size=0.2, random_state=42)
val_images, test_images, val_annot, test_annot = train_test_split(temp_images, temp_annot, test_size=0.25, random_state=42)  # 0.2 * 0.25 = 0.05

# Function to process a split
def process_split(images, annots, split_name):
    mask_split_dir = os.path.join(mask_dir, split_name)
    image_split_dir = os.path.join(image_dir, split_name)
    label_split_dir = os.path.join(label_dir, split_name)
    os.makedirs(mask_split_dir, exist_ok=True)
    os.makedirs(image_split_dir, exist_ok=True)
    os.makedirs(label_split_dir, exist_ok=True)
    
    for img_path, annot_path in zip(images, annots):
        # Copy image and annotation
        shutil.copy(img_path, os.path.join(image_split_dir, os.path.basename(img_path)))
        shutil.copy(annot_path, os.path.join(label_split_dir, os.path.basename(annot_path)))
        
        img = mpimg.imread(img_path)
        annots_data = load_annotations(annot_path)
        
        masks = []
        bboxes = []
        class_ids = []
        for annot in annots_data:
            mask, bbox, class_id = create_mask(img, annot)
            masks.append(mask)
            bboxes.append(bbox)
            class_ids.append(int(class_id))
        
        # Save masks
        for i, (mask, bbox, class_id) in enumerate(zip(masks, bboxes, class_ids)):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_split_dir, f'{base_name}_mask_{i}_class{class_id}.png')
            mpimg.imsave(mask_path, mask, cmap='gray')

# Process each split
print("Processing train split...")
process_split(train_images, train_annot, 'train')
print("Processing validation split...")
process_split(val_images, val_annot, 'val')
print("Processing test split...")
process_split(test_images, test_annot, 'test')

print("Mask, image, and label generation complete!")