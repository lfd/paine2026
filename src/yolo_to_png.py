import os
import cv2
import numpy as np

def yolo_to_filled_masks(labels_dir, images_dir, output_dir):
    """
    YOLOv8 Bounding Boxes -> PNG/JPG Mask 

    Foreground = 255
    Background = 0
    """

    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(label_file)[0]
        label_path = os.path.join(labels_dir, label_file)

        # passendes Bild suchen
        image_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"No image found for {label_file}")
            continue

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))

                # YOLO: class x_center y_center width height
                _, x_center, y_center, bw, bh = parts

                # -> Pixelkoordinaten
                x_center *= w
                y_center *= h
                bw *= w
                bh *= h

                x1 = int(x_center - bw / 2)
                y1 = int(y_center - bh / 2)
                x2 = int(x_center + bw / 2)
                y2 = int(y_center + bh / 2)

                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                
                mask[y1:y2, x1:x2] = 255

        output_path = os.path.join(output_dir, base_name + ".jpg")
        cv2.imwrite(output_path, mask)

        print(f"{output_path} saved")


# Beispiel
if __name__ == "__main__":
    yolo_to_filled_masks(
        labels_dir="./data/raw/pin_pcb_detection_roboflow/train/labels",
        images_dir="./data/raw/pin_pcb_detection_roboflow/train/images",
        output_dir="./data/training/masks"
    )