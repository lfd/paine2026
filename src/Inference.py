# Inference script for segmentation using a U-Net model
# for individual images or folders containing images
# runs prediction, saves prediction masks and optionally computes metrics if Ground Truth is available

# Configuration: set paths, Model names, Patch size, Batch size, etc. (as seen below)

import os
import math
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from fastai.vision.all import *

# ========== Configuration ==========
INPUT_PATH     = "./data/test/images"                       # folder or single image path for inference (e.g., "./input" or "./input/image1.png")
OUTPUT_DIR     = "./output/predicted_masks"                 # output folder for saving prediction masks (will be created if it doesn't exist)
SKIP_PNG_SAVE  = True                                       # True = no intermediate PNGs saved, False = save original image as PNG in output folder
MODEL_NAME     = "unet_trained"                             # Name without .pth extension (e.g., "unet" for unet.pth)
MODEL_DIR      = './notebooks/models'                       # folder, where the model is saved
PATCH_SIZE     = (512, 512)                                 # Patch Size (H, W)
NUM_CLASSES    = 2                                          # number of classes as in training (n_out=2)
BATCH_SIZE     = 8                                          # Batch Size for inference (adjust based on GPU memory)
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
GT_MASK_PATH   = None                                       # Optional: Ground-Truth-Mask
OVERLAY_ALPHA  = 0.35                                       # Transparency for overlaying prediction mask on original image (0.0 = fully transparent, 1.0 = fully opaque)
# ===================================

# ------------------------------
# Helping functions: I/O, Patches, Stitching
# ------------------------------

def list_image_files(input_path):
    if os.path.isdir(input_path):
        exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(exts)
        )
        if not files:
            raise FileNotFoundError(f"No Image Data found in: {input_path}")
        return files
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(f"Input Path not found: {input_path}")


def load_image_as_gray(image_path, save_png_path=None):
    img = Image.open(image_path).convert("L")  
    if save_png_path:
        img.save(save_png_path)
    return np.array(img)  


def pad_to_multiple(img_np, patch_size):
    H, W = img_np.shape
    ph, pw = patch_size
    H_pad = math.ceil(H / ph) * ph
    W_pad = math.ceil(W / pw) * pw
    pad_h = H_pad - H
    pad_w = W_pad - W
    img_pad = np.pad(img_np, ((0, pad_h), (0, pad_w)), mode="reflect")
    return img_pad, (H, W)


def extract_patches(img_np, patch_size):
    ph, pw = patch_size
    H, W = img_np.shape
    patches, coords = [], []
    for y in range(0, H, ph):
        for x in range(0, W, pw):
            patch = img_np[y:y+ph, x:x+pw]
            patches.append(patch)
            coords.append((y, x))
    return patches, coords, (H, W)


def reconstruct_from_patches(patches, coords, full_shape):
    H, W = full_shape
    canvas = np.zeros((H, W), dtype=patches[0].dtype)
    ph, pw = patches[0].shape
    for patch, (y, x) in zip(patches, coords):
        canvas[y:y+ph, x:x+pw] = patch
    return canvas


def colorize_multiclass(mask_np, num_classes):
    cmap = plt.get_cmap('tab20')
    colored = cmap((mask_np % num_classes) / max(num_classes-1, 1))
    return (colored[..., :3] * 255).astype(np.uint8)


def compute_metrics(pred_mask_np, gt_mask_np, num_classes=2):
    assert pred_mask_np.shape == gt_mask_np.shape
    dices, ious = [], []
    for c in range(num_classes):
        pred_c = (pred_mask_np == c)
        gt_c   = (gt_mask_np == c)
        inter  = np.logical_and(pred_c, gt_c).sum()
        union  = np.logical_or(pred_c, gt_c).sum()
        iou_c  = inter / (union + 1e-8)
        dice_c = (2 * inter) / (pred_c.sum() + gt_c.sum() + 1e-8)
        ious.append(iou_c)
        dices.append(dice_c)
    acc = (pred_mask_np == gt_mask_np).mean()
    return {"accuracy": acc, "mIoU": float(np.mean(ious)), "mDice": float(np.mean(dices))}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_model():
    dummy_x = torch.zeros(1, 3, PATCH_SIZE[0], PATCH_SIZE[1])
    dummy_y = torch.zeros(1, PATCH_SIZE[0], PATCH_SIZE[1]).long()
    dls = DataLoaders.from_dsets([(dummy_x[0], dummy_y[0])], [(dummy_x[0], dummy_y[0])], bs=1, device=DEVICE)
    learn = unet_learner(dls, resnet34, n_out=NUM_CLASSES, pretrained=False,
                         loss_func=CrossEntropyLossFlat(axis=1))
    learn.model_dir = MODEL_DIR
    learn.load(MODEL_NAME)
    return learn.model.eval().to(DEVICE)


def infer_image(image_path, save_png_path, save_pred_path, model, mean, std):
    img_np = load_image_as_gray(image_path, save_png_path)
    H_orig, W_orig = img_np.shape
    img_pad, orig_shape = pad_to_multiple(img_np, PATCH_SIZE)
    patches, coords, full_shape = extract_patches(img_pad, PATCH_SIZE)
    print(f"{os.path.basename(image_path)}: Original size {img_np.shape}, Patches {len(patches)}, padded size{full_shape}")

    preds_list = []
    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            batch_np = patches[i:i+BATCH_SIZE]
            batch_3ch = [np.stack([p]*3, axis=-1) for p in batch_np]
            batch_t = torch.stack([torch.from_numpy(p).permute(2,0,1) for p in batch_3ch]).float() / 255.0
            batch_t = ((batch_t.to(DEVICE) - mean) / std)
            logits = model(batch_t)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.extend(pred)

    pred_canvas = reconstruct_from_patches(preds_list, coords, full_shape)
    pred_final = pred_canvas[:H_orig, :W_orig]
    colored = colorize_multiclass(pred_final, NUM_CLASSES)
    Image.fromarray(colored).save(save_pred_path)
    print(f"Prediction saved in: {save_pred_path}")

    if GT_MASK_PATH and os.path.exists(GT_MASK_PATH):
        gt = np.array(Image.open(GT_MASK_PATH).convert("L"))
        gt = gt[:H_orig, :W_orig]
        metrics = compute_metrics(pred_final.astype(np.int32), gt.astype(np.int32), num_classes=NUM_CLASSES)
        print(f"  Pixel-Acc: {metrics['accuracy']:.4f} | mIoU: {metrics['mIoU']:.4f} | mDice: {metrics['mDice']:.4f}")


def main():
    print(f"Device: {DEVICE}")

    input_files = list_image_files(INPUT_PATH)
    ensure_dir(OUTPUT_DIR)

    model = prepare_model()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)

    for image_path in input_files:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_png_path = None if SKIP_PNG_SAVE else os.path.join(OUTPUT_DIR, f"{stem}.png")
        save_pred_path = os.path.join(OUTPUT_DIR, f"{stem}_prediction_mask.png")
        infer_image(image_path, save_png_path, save_pred_path, model, mean, std)

if __name__ == "__main__":
    main()
