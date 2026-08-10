# paine2026
Reproduction Package

## Project Structure

main/
    notebooks/
        Segmentation.ipynb
        Inference.ipynb
        Model_evaluation.ipynb
        logreg.ipynb
    src/
        Images_to_Patches.py
        JSON_to_PNG.py
        Inference.py
        pinarea_decision.py
    models/
        board_passfail_model.joblib
        

## Required Packages
All dependencies can be found in the file:
requirements2.txt

Install:
pip install -r requirements.txt

## Models & Data
Due to confidentiality agreements with the industrial project partner,
the PCB image data cannot be released publicly. The repository contains
all source code, model configurations, and evaluation scripts required
to reproduce the reported methodology.
The cited Roboflow data can be accessed here: https://universe.roboflow.com/pcbpindetection/pin_pcb_detection

## Reproduction Steps

1. Install Dependencies
pip install -r requirements.txt

2. Download or save your training data 

3. If necessary, annotate the data (e.g. in Labelme): save each mask as a JSON file with the same filename
    Annotation: polygon masks for the pins (alternatively: bounding boxes around the pins)

Proposed Method:

    Training:
    1. JSON_to_PNG.py converts JSON templates (from Labelme) 
    2. Images_to_Patches.py splits input PNG masks into PNG patches and converts input .bmp files into PNG patches
    3. Use Segmentation.ipynb for training:
        Input:  - Specify the training data folder containing the subfolders “images” and “masks”
                - The filenames of the image and mask must match
        Output: - Trained UNet model as a .pth file
    4. logreg.ipynb trains a decision model
        Input: CSV file containing features and ground truth labels (e.g. from pinarea_decision.py using a former joblib file (one can be found in this repository))
        Output: trained decision model as a joblib file

    Inference/Application:
    1. If necessary, evaluate the model on annotated test data in model_evaluation.py
    2. Use Inference.ipynb to make predictions on an image (for testing purposes):
        Input:  - a .bmp or .png image
                - Model/weights as .pth
        Output: - Image as PNG
                - Mask as PNG
                - Visualisations in the notebook
    3. Use Inference.py for prediction
        Input: individual images or folders
        Output: predicted masks as PNG
    4. pinarea_decision.py: contour detection, area calculation of contours
        Input:  - (predicted) PNG masks
        Output: - CSV with features and decision
        
