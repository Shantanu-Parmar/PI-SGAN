import numpy as np
from scipy.optimize import curve_fit
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define 2D Gaussian function for PSF fitting
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

# Function to load YOLO label and return bounding boxes with class IDs
def load_yolo_label(label_path):
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, xc, yc, w, h = map(float, line.strip().split())
            bboxes.append((int(class_id), xc, yc, w, h))
    return bboxes

# Function to convert YOLO bbox to pixel coords
def yolo_to_pixels(xc, yc, w, h, img_width, img_height):
    x_min = int((xc - w / 2) * img_width)
    y_min = int((yc - h / 2) * img_height)
    x_max = int((xc + w / 2) * img_width)
    y_max = int((yc + h / 2) * img_height)
    return x_min, y_min, x_max, y_max

# Function to fit PSF on a crop
def fit_psf(crop):
    # Create meshgrid
    x = np.linspace(0, crop.shape[1] - 1, crop.shape[1])
    y = np.linspace(0, crop.shape[0] - 1, crop.shape[0])
    x, y = np.meshgrid(x, y)

    # Initial guess
    initial_guess = (np.max(crop), crop.shape[1]/2, crop.shape[0]/2, 3, 3, 0)

    # Fit
    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y), crop.ravel(), p0=initial_guess)
        return popt
    except:
        return None

# Main function to process dataset
def extract_psf_from_dataset(images_dirs, labels_dirs, output_dir='psf_fits', class_ids=[2, 3, 4, 5, 6, 7]):  # Single star classes, exclude clusters (0) and planets (1) if not point-like
    os.makedirs(output_dir, exist_ok=True)
    psf_params_list = []

    for split in ['train', 'val', 'test']:
        images_path = os.path.join(images_dirs, split)
        labels_path = os.path.join(labels_dirs, split)

        for img_file in os.listdir(images_path):
            if not img_file.endswith('.jpg'):
                continue

            img_path = os.path.join(images_path, img_file)
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(labels_path, label_file)

            if not os.path.exists(label_path):
                continue

            # Load image as grayscale
            image = np.array(Image.open(img_path).convert('L'))
            img_height, img_width = image.shape

            # Load bboxes
            bboxes = load_yolo_label(label_path)

            for class_id, xc, yc, w, h in bboxes:
                if class_id not in class_ids:  # Skip non-point sources
                    continue

                x_min, y_min, x_max, y_max = yolo_to_pixels(xc, yc, w, h, img_width, img_height)
                crop = image[y_min:y_max, x_min:x_max]

                if crop.size == 0:
                    continue

                # Fit PSF
                popt = fit_psf(crop)
                if popt is not None:
                    psf_params_list.append(popt)
                    # Save plot
                    fitted_psf = gaussian_2d((np.meshgrid(np.arange(crop.shape[1]), np.arange(crop.shape[0]))), *popt).reshape(crop.shape)
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].imshow(crop, cmap='gray')
                    axs[0].set_title('Crop')
                    axs[1].imshow(fitted_psf, cmap='gray')
                    axs[1].set_title('Fitted PSF')
                    plt.savefig(os.path.join(output_dir, f'psf_{split}_{img_file}_{class_id}.png'))
                    plt.close()

    # Average PSF params
    if psf_params_list:
        avg_params = np.mean(psf_params_list, axis=0)
        print(f"Average PSF Parameters: Amplitude={avg_params[0]:.2f}, Center=({avg_params[1]:.2f}, {avg_params[2]:.2f}), Sigma=({avg_params[3]:.2f}, {avg_params[4]:.2f}), Theta={avg_params[5]:.2f}")
    else:
        print("No valid PSF fits found.")

# Run
images_dirs = 'Images'
labels_dirs = 'Labels'
extract_psf_from_dataset(images_dirs, labels_dirs)
