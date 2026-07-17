# Code for converting BMP images to greyscale PNG images and for splitting these images into patches (works if already PNG)
# Configuration: Setting the input and output folders, as well as the patch size and overlap

import os
from PIL import Image

# === Configuration ===
input_dir = "./images"                              # folder containing BMP or PNG images
output_dir_png = "./png_images"                     # folder for converted PNG images (if needed)
output_dir_patches = "./output"                     # folder for the resulting patches
patch_size = (512, 512)                             # Patch size (W, H)
overlap = 0                                         # optional: overlap between patches (in pixels) 

##################################################################################

# create folders if they don't exist
os.makedirs(output_dir_png, exist_ok=True)
os.makedirs(output_dir_patches, exist_ok=True)

# BMP -> PNG (Grayscale)
def convert_to_png_grayscale(image_path, output_path):
    img = Image.open(image_path)
    img_gray = img.convert("L")  
    img_gray.save(output_path, "PNG")
    return img_gray

# image to patches
def split_into_patches(img, base_name, patch_size, overlap, output_dir):
    width, height = img.size
    patch_width, patch_height = patch_size

    step_x = patch_width - overlap
    step_y = patch_height - overlap

    patch_count = 0
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            box = (x, y, min(x + patch_width, width), min(y + patch_height, height))
            patch = img.crop(box)

            # Padding, if the patch is smaller than the desired size
            if patch.size != patch_size:
                padded_patch = Image.new("L", patch_size, color=0)
                padded_patch.paste(patch, (0, 0))
                patch = padded_patch

            patch_filename = f"{base_name}_patch_{patch_count}.png"
            patch.save(os.path.join(output_dir, patch_filename))
            patch_count += 1

    return patch_count


# main process: BMP -> PNG + Patches
# comment, if only patches are needed

"""# main process
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".bmp"):
        bmp_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        # convert
        png_path = os.path.join(output_dir_png, f"{base_name}.png")
        img_gray = convert_to_png_grayscale(bmp_path, png_path)

        # create patches
        num_patches = split_into_patches(img_gray, base_name, patch_size, overlap, output_dir_patches)
        print(f"Verarbeitet: {filename} -> {num_patches} Patches created.")"""

# main process: only Patches
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        img = Image.open(png_path)
        # create paatches
        num_patches = split_into_patches(img, base_name, patch_size, overlap, output_dir_patches)
        print(f"Verarbeitet: {filename} -> {num_patches} Patches created.")

