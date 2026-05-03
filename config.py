# config.py
# Central configuration for hyperparameters, paths, and image settings.

import os

# Training hyperparameters
batchsize = 16
epochs = 10
lr = 1e-4
train_split = 0.80
random_seed = 42

# Image dimensions
resize_x = 224
resize_y = 224
input_channels = 3

# Dataset / model
num_classes = 4
class_names = ["car", "bus", "truck", "motorcycle"]

# COCO category IDs for the four selected vehicle classes.
COCO_CLASS_MAP = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRAIN_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")
EVAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_results.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "project_report.md")
VISUALS_DIR = os.path.join(RESULTS_DIR, "segmentation_outputs")

# Segmentation / detection settings
SEG_CONF_THRESHOLD = 0.40
MIN_BOX_AREA = 500
