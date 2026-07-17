# Code for deciding pass/fail based on segmentation mask statistics and a pre-trained model
# The script processes either a single image or all images in a specified folder

# Configuration: Set input path (file or folder), output directory, model artifact path, and supported extensions
# Output is a CSV file with statistics and decisions for each processed image

import os
import math
import cv2
import numpy as np
import joblib
import pandas as pd
from datetime import datetime

# ====== Configuration (edit these) ======
INPUT_PATH = "./predicted_masks"   # folder or single image path
OUTPUT_DIR = "./output"
MODEL_ARTIFACT = "board_passfail_model.joblib"
EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
# ========================================


def list_image_files(input_path):
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(EXTS)
        )
        return files
    if os.path.isfile(input_path) and input_path.lower().endswith(EXTS):
        return [input_path]
    return []


def compute_board_stats_from_mask(mask_gray):
    # mask_gray: single-channel numpy array (grayscale prediction mask)
    _, thresh = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_lst = []
    ratio_data = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area == 0:
            continue
        area_lst.append(area)

        rect = cv2.minAreaRect(cnt)
        (rect_x, rect_y), (rect_w, rect_h), angle = rect
        long_side = max(rect_w, rect_h)
        short_side = min(rect_w, rect_h)
        if short_side > 0:
            true_aspect_ratio = float(long_side) / short_side
        else:
            true_aspect_ratio = 1.0
        ratio_data.append(true_aspect_ratio)

    if not area_lst:
        return {
            "avg_flaeche": 0.0,
            "std_flaeche": 0.0,
            "max_flaeche": 0.0,
            "anzahl_pins": 0,
            "mean_aspect_ratio": 1.0,
        }

    avg_area = sum(area_lst) / len(area_lst)
    if len(area_lst) > 1:
        variance = sum((x - avg_area) ** 2 for x in area_lst) / (len(area_lst) - 1)
        std_area = math.sqrt(variance)
    else:
        std_area = 0.0

    mean_aspect_ratio = sum(ratio_data) / len(ratio_data) if ratio_data else 1.0

    return {
        "avg_flaeche": float(avg_area),
        "std_flaeche": float(std_area),
        "max_flaeche": float(max(area_lst)),
        "anzahl_pins": int(len(area_lst)),
        "mean_aspect_ratio": float(mean_aspect_ratio),
    }


def load_model(artifact_path):
    art = joblib.load(artifact_path)
    return art


def decide_board(artifact, board_stats_row: pd.Series, margin=0.02):
    model = artifact["model"]
    THRESHOLD_PASS = artifact["threshold_pass"]
    feature_cols = artifact["feature_cols"]

    X = pd.DataFrame([board_stats_row[feature_cols]])
    X = X.fillna(0.0)
    p_pass = float(model.predict_proba(X)[0, 1])
    label = "pass" if p_pass >= THRESHOLD_PASS else "fail"
    if abs(p_pass - THRESHOLD_PASS) < margin:
        label = "review"
    return label, p_pass


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = list_image_files(INPUT_PATH)
    if not files:
        print(f"No images found in {INPUT_PATH}")
        return

    artifact = load_model(MODEL_ARTIFACT)

    results = []
    for p in files:
        print(f"Processing: {p}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Skipped (cannot read): {p}")
            continue

        stats = compute_board_stats_from_mask(img)
        row = pd.Series(stats)
        label, p_pass = decide_board(artifact, row)

        ts = datetime.now().isoformat()
        result = {
            "file": os.path.basename(p),
            "timestamp": ts,
            **stats,
            "label": label,
            "p_pass": float(p_pass),
        }
        results.append(result)
        print(f"  -> {result['label']} (P(pass)={result['p_pass']:.3f})")

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, f"decision_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Results written to: {out_csv}")


if __name__ == "__main__":
    main()
