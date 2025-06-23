import argparse
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import csv

def normalize_band(band):
    lo, hi = np.nanpercentile(band, [2, 98])
    return np.clip((band - lo) / (hi - lo), 0, 1)

def save_rgb_image(img, bands, output_dir, gamma=2.2):
    red, green, blue = [img[b-1] for b in bands]
    rgb = np.stack([
        normalize_band(red),
        normalize_band(green),
        normalize_band(blue)
    ], axis=-1)
    # rgb = np.power(rgb, 1 / gamma)

    output_path = os.path.join(output_dir, "rgb.jpeg")
    plt.imsave(output_path, rgb)
    print(f"Saved RGB image as: {output_path}")

def save_greyscale_band_and_csv(band, band_index, output_dir):
    # Save grayscale JPEG
    norm_band = normalize_band(band)
    output_img = os.path.join(output_dir, f"band-{band_index}.jpeg")
    plt.imsave(output_img, norm_band, cmap='gray')
    print(f"Saved grayscale image as: {output_img}")

    # Save CSV
    output_csv = os.path.join(output_dir, f"band-{band_index}.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in band:
            writer.writerow(row)
    print(f"Saved pixel values as CSV: {output_csv}")

def print_band_info(src):
    print(f"\n=== Band Information for {src.name} ===")
    for i in range(1, src.count + 1):
        desc = src.descriptions[i-1] if src.descriptions[i-1] else "Unknown"
        shape = (src.height, src.width)
        print(f"Band {i}: Description: {desc} | Shape: {shape}")

def find_rgb_bands(src):
    return [4, 3, 2]
    band_indices = {}
    for i, desc in enumerate(src.descriptions):
        if desc:
            desc_clean = desc.strip().upper()
            if desc_clean == 'B2':
                band_indices['B2'] = i + 1
            elif desc_clean == 'B3':
                band_indices['B3'] = i + 1
            elif desc_clean == 'B4':
                band_indices['B4'] = i + 1
    if all(b in band_indices for b in ['B2', 'B3', 'B4']):
        return [band_indices['B4'], band_indices['B3'], band_indices['B2']]
    return None

def main():
    parser = argparse.ArgumentParser(description="TIF Band Info and Export Tool")
    parser.add_argument('tif_file', help='Path to the TIF file')
    parser.add_argument('--band', type=int, help='Optional band number for grayscale and CSV export')
    args = parser.parse_args()

    output_dir = "./dump"
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(args.tif_file) as src:
        img = src.read()

        if args.band:
            if 1 <= args.band <= src.count:
                band = img[args.band - 1]
                save_greyscale_band_and_csv(band, args.band, output_dir)
            else:
                print(f"Invalid band number {args.band}. File has {src.count} bands.")
        else:
            print_band_info(src)
            rgb_bands = find_rgb_bands(src)
            if rgb_bands:
                save_rgb_image(img, rgb_bands, output_dir)
            else:
                print("\nRGB bands (B2, B3, B4) not found. Skipping RGB image generation.")

if __name__ == "__main__":
    main()
