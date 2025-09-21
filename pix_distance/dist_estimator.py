import os
import logging
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

# Configuration
IMAGES_DIR = "images/train"  # Directory with images
LABELS_DIR = "labels/train"  # Directory with label files
CLASS_NAMES = ["Pleiades", "Jupiter", "Betelgeuse", "Aldebaran", "Zeta_Tauri", "Elnath", "Hassaleh", "Bellatrix"]
BETELGEUSE_ID = 2  # Betelgeuse class ID
BELLATRIX_ID = 7   # Bellatrix class ID

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("distance_log.txt"), logging.StreamHandler()])
logger = logging.getLogger()

def compute_pixel_distance(images_dir, labels_dir):
    distances = []
    coords = {"Betelgeuse": [], "Bellatrix": []}
    for image_file in os.listdir(images_dir):
        if not image_file.endswith(".jpg"):
            continue
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            logger.warning(f"Label file {label_file} not found for {image_file}, skipping")
            continue
        
        # Get image size
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Parse labels
        betelgeuse_pos = None
        bellatrix_pos = None
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                if class_id == BETELGEUSE_ID:
                    betelgeuse_pos = (x_center, y_center)
                elif class_id == BELLATRIX_ID:
                    bellatrix_pos = (x_center, y_center)
        
        if betelgeuse_pos and bellatrix_pos:
            dx = betelgeuse_pos[0] - bellatrix_pos[0]
            dy = betelgeuse_pos[1] - bellatrix_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
            coords["Betelgeuse"].append(betelgeuse_pos)
            coords["Bellatrix"].append(bellatrix_pos)
            logger.info(f"Image {image_file}: Betelgeuse at ({betelgeuse_pos[0]:.1f}, {betelgeuse_pos[1]:.1f}), Bellatrix at ({bellatrix_pos[0]:.1f}, {bellatrix_pos[1]:.1f}), Distance: {distance:.1f} pixels")
        else:
            logger.warning(f"Missing Betelgeuse or Bellatrix in {label_file}")
    
    if distances:
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        logger.info(f"Average pixel distance across {len(distances)} images: {avg_distance:.1f} ± {std_distance:.1f}")
        
        # Save to CSV
        with open("distances.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Betelgeuse_X", "Betelgeuse_Y", "Bellatrix_X", "Bellatrix_Y", "Distance_Pixels"])
            for i, (img, dist) in enumerate(zip(os.listdir(images_dir), distances)):
                if img.endswith(".jpg"):
                    label_file = os.path.splitext(img)[0] + ".txt"
                    label_path = os.path.join(labels_dir, label_file)
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            bet_x, bet_y, bell_x, bell_y = None, None, None, None
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    class_id = int(parts[0])
                                    x_center = float(parts[1]) * img_width  # Reuse last img_width (simplified)
                                    y_center = float(parts[2]) * img_height
                                    if class_id == BETELGEUSE_ID:
                                        bet_x, bet_y = x_center, y_center
                                    elif class_id == BELLATRIX_ID:
                                        bell_x, bell_y = x_center, y_center
                            if bet_x is not None and bell_x is not None:
                                writer.writerow([img, bet_x, bet_y, bell_x, bell_y, dist])
        logger.info("Distances saved to distances.csv")
        
        # Visualization: Bar Chart of Distance Distribution
        distance_bins = np.histogram_bin_edges(distances, bins='auto')
        distance_counts, _ = np.histogram(distances, bins=distance_bins)
        plt.figure(figsize=(10, 6))
        plt.bar([f"{int(b)}-{int(e)}" for b, e in zip(distance_bins[:-1], distance_bins[1:])], distance_counts, color="#4BC0C0", edgecolor="#36A2EB")
        plt.title("Distribution of Pixel Distances")
        plt.xlabel("Pixel Distance")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("distance_distribution.png")
        plt.close()

        # Visualization: Scatter Plot of Coordinates
        bet_x = [c[0] for c in coords["Betelgeuse"]]
        bet_y = [c[1] for c in coords["Betelgeuse"]]
        bell_x = [c[0] for c in coords["Bellatrix"]]
        bell_y = [c[1] for c in coords["Bellatrix"]]
        plt.figure(figsize=(10, 6))
        plt.scatter(bet_x, bet_y, color="#FF6384", label="Betelgeuse", s=50)
        plt.scatter(bell_x, bell_y, color="#36A2EB", label="Bellatrix", s=50)
        plt.title("Betelgeuse vs Bellatrix Coordinates")
        plt.xlabel("X Coordinate (pixels)")
        plt.ylabel("Y Coordinate (pixels)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("coordinates_scatter.png")
        plt.close()
    else:
        logger.error("No valid distances found—ensure both objects are labeled in images")

if __name__ == "__main__":
    compute_pixel_distance(IMAGES_DIR, LABELS_DIR)