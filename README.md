## paine2026
This reproduction package accompanies the paper
"Quality Inspection of Printed Circuit Board Pin Insertion via Semantic Segmentation and Board-Level Feature Extraction".

## Overview

This reproduction package accompanies the paper
"Quality Inspection of Printed Circuit Board Pin Insertion via Semantic Segmentation and Board-Level Feature Extraction".

The proposed approach combines semantic segmentation
with a decision module for automated pin defect analysis
of printed circuit board (PCB) images.

The package contains:

- data preprocessing utilities
- segmentation model training and evaluation notebooks
- inference scripts for generating prediction masks
- a trained decision classifier
- scripts required to reproduce the workflow

A subset of the publicly available dataset used in the
paper is included. The proprietary industrial dataset
described in the paper cannot be distributed.

The package demonstrates the inference and decision 
workflow proposed in the paper. Training and evaluation 
notebooks are included, while the automated 
reproduction workflow uses the provided pretrained models.

## Models & Data
Due to confidentiality agreements with the industrial project partner,
the PCB image data cannot be released publicly. The repository contains
all source code, model configurations, and evaluation scripts required
to reproduce the reported methodology.
The cited Roboflow data is shared in this repository and can be accessed here: 
https://universe.roboflow.com/pcbpindetection/pin_pcb_detection

License of the used Roboflow data: CC BY 4.0

# Download pretrained models

The pretrained model weights are available at:

https://doi.org/10.5281/zenodo.21887281

Download the file and place it into:

```text
notebooks/
    models/
        unet_trained.pth
```

## Project Structure

main/  
    README.md  
    requirements.txt  
    dockerfile.reproduction  
    run.sh  
    .gitignore  
    .devcontainer  
    notebooks/  
        Segmentation.ipynb  
        Inference.ipynb  
        Model_evaluation.ipynb  
        logreg.ipynb  
        models  
    src/  
        Images_to_Patches.py  
        JSON_to_PNG.py  
        Inference.py  
        pinarea_decision.py  
    models/  
        board_passfail_model.joblib  
    data/  
        raw/pin_pcb_detection_roboflow  
            test  
                images  
                labels  
            train  
                images   
                labels  
            valid  
                images  
                labels  
        test  
            images  
            masks  
        training  
            images  
            masks  
    output/  
        decision_tables  
        predicted_masks  


        

## Required Packages
All dependencies can be found in the file:
requirements.txt

Install:
```bash
pip install -r requirements.txt
```

Important are the versions of these packages (as in requirements.txt):  
fastai==2.8.6  
fastcore==1.12.11  
torch==2.9.1  
torchvision==0.24.1  

## Automated Reproduction

Run in Docker:
```bash
docker build -t paine2026 -f dockerfile.reproduction .
docker --rm run paine2026
```

or run:
```bash
bash run.sh
```

The script performs:

1. Segmentation inference.
2. Generation of prediction masks.
3. Execution of the decision function.
4. Creation of the final CSV output.

## Expected Outputs

output/  
    predicted_masks  
        001prediction_mask.png  
        002prediction_mask.png  
        ...  
    decision_tables  
        decision_results.csv  


## Full Workflow Steps

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
        
