#!/bin/bash

set -e

echo "Running inference on test image"
python src/Inference.py

echo "Running decision table generation"
python src/pinarea_decision.py

echo "Finished."