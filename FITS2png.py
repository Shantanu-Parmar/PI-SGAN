import os
import numpy as np
from astropy.io import fits
from PIL import Image

# === Configuration ===
FITS_DIR = "Ground_truths"  # Source directory
OUTPUT_DIR = "MobilTelesco_Processed/reference_images"
TARGET_SIZE = (800, 800)  # Output PNG size

# === Create base output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Process each FITS file ===
for filename in os.listdir(FITS_DIR):
    if filename.lower().endswith(".fits"):
        fits_path = os.path.join(FITS_DIR, filename)

        # Extract object name from the beginning of the filename
        obj_name = filename.split('_')[0]
        obj_output_dir = os.path.join(OUTPUT_DIR, obj_name)
        os.makedirs(obj_output_dir, exist_ok=True)

        try:
            with fits.open(fits_path) as hdul:
                for hdu_index, hdu in enumerate(hdul):
                    data = hdu.data
                    if data is None:
                        continue  # Skip HDUs without image data
                data = np.nan_to_num(hdu.data)

                # === RGB Image (3-channel or 4-channel, use first 3 planes) ===
                if data.ndim == 3 and data.shape[0] in [3, 4]:
                    rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
                    for i in range(3):
                        channel = data[i]
                        ch_min = np.min(channel)
                        ch_max = np.max(channel)
                        if ch_max > ch_min:
                            norm = 255 * (channel - ch_min) / (ch_max - ch_min)
                        else:
                            norm = np.zeros_like(channel)
                        rgb[..., i] = norm.astype(np.uint8)

                    img = Image.fromarray(rgb, mode='RGB')
                    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

                    output_name = f"{os.path.splitext(filename)[0]}_rgb.png"
                    output_path = os.path.join(obj_output_dir, output_name)
                    img.save(output_path)
                    print(f"✅ Saved RGB image: {output_path}")

                # === Grayscale image ===
                elif data.ndim == 2:
                    d_min = np.min(data)
                    d_max = np.max(data)
                    if d_max > d_min:
                        norm = 255 * (data - d_min) / (d_max - d_min)
                    else:
                        norm = np.zeros_like(data)
                    norm = norm.astype(np.uint8)

                    img = Image.fromarray(norm)
                    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

                    output_name = f"{os.path.splitext(filename)[0]}_gray.png"
                    output_path = os.path.join(obj_output_dir, output_name)
                    img.save(output_path)
                    print(f"✅ Saved grayscale image: {output_path}")

                else:
                    print(f"⚠️ Skipped {filename} (unexpected shape: {data.shape})")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print("\n🎉 All FITS files converted to PNGs successfully.")
