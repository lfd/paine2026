#!/bin/bash

set -e

if [ ! -f notebooks/models/unet_trained.pth ]; then
    wget <https://zenodo.org/records/21887282/files/unet_trained.pth> -O notebooks/models/unet_trained.pth
fi

echo "Running inference on test image"
python src/Inference.py

echo "Running decision table generation"
python src/pinarea_decision.py

echo "Finished."