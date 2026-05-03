# Vehicle Segmentation and Classification

This project implements a two-stage vehicle recognition pipeline:

1. Detect or segment vehicles in an input image with a COCO Mask R-CNN model.
2. Classify each detected vehicle crop as `car`, `bus`, `truck`, or `motorcycle` with a ResNet-18 classifier.

The repository includes the dataset, a trained checkpoint, evaluation outputs, and example segmentation results.

## Project Structure

```text
.
+-- config.py                         # Shared paths, classes, and hyperparameters
+-- dataset.py                        # VehicleDataset and dataloader helpers
+-- model.py                          # ResNet-18 based 4-class classifier
+-- train.py                          # Training and classifier evaluation script
+-- predict.py                        # Detection, segmentation, and prediction pipeline
+-- evaluate.py                       # Final evaluation, demo prediction, and report generation
+-- interface.py                      # Grader-facing exported names
+-- data/                             # Image dataset grouped by vehicle class
+-- checkpoints/final_weights.pth     # Trained classifier weights
+-- results/                          # Metrics, generated report, and visual outputs
```

## Dataset

Images are stored in class folders under `data/`:

```text
data/
+-- Car/
+-- Bus/
+-- Truck/
+-- motorcycle/
```

The dataset contains 400 total images:

| Class | Images |
| --- | ---: |
| car | 100 |
| bus | 100 |
| truck | 100 |
| motorcycle | 100 |

`VehicleDataset` can infer labels either from the parent folder name or from a class-name filename prefix.

## Model

The classifier uses:

- Backbone: `torchvision.models.resnet18`
- Input size: `224 x 224`
- Classes: `car`, `bus`, `truck`, `motorcycle`
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 10
- Batch size: 16
- Learning rate: `1e-4`

For full-image prediction, `predict.py` uses `torchvision` Mask R-CNN with COCO weights to find vehicle regions, then sends each crop through the trained classifier.

## Setup

Create and activate a Python environment, then install the required packages:

```bash
pip install torch torchvision pillow tqdm
```

If you are using a GPU, install the PyTorch build that matches your CUDA version from the official PyTorch installation selector.

## Training

Run:

```bash
python train.py
```

This will:

- Load images from `data/`
- Create a stratified 80/20 train-test split
- Train the classifier
- Save weights to `checkpoints/final_weights.pth`
- Save training history to `results/training_history.json`

Training settings are controlled in `config.py`.

## Evaluation

Run:

```bash
python evaluate.py
```

This will:

- Load the trained classifier weights
- Evaluate on the held-out test split
- Run the full detection and classification pipeline on sample images
- Save metrics to `results/evaluation_results.json`
- Save a report to `results/project_report.md`
- Save annotated outputs and crops to `results/segmentation_outputs/`

Current recorded classifier performance:

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.9250 |
| Test loss | 0.2197 |
| Car accuracy | 0.9500 |
| Bus accuracy | 0.9000 |
| Truck accuracy | 0.8500 |
| Motorcycle accuracy | 1.0000 |

## Prediction

Predict on one or more images:

```bash
python predict.py path/to/image.jpg
```

Save annotated segmentation outputs and vehicle crops:

```bash
python predict.py path/to/image.jpg --save
```

If no image path is provided, `predict.py` tries to run on one sample image from each vehicle class in `data/`.

The predictor returns JSON-style results containing:

- Source image path
- Number of detections
- Detector label and confidence
- Bounding box coordinates
- Final classifier label and confidence
- Output paths when `--save` is used

## Grader Interface

`interface.py` exports standard names:

```python
the_batch_size
total_epochs
TheDataset
the_dataloader
TheModel
the_predictor
the_trainer
```

These aliases point to the implementation in `config.py`, `dataset.py`, `model.py`, `predict.py`, and `train.py`.

## Notes

- The first run of `model.py`, `train.py`, `predict.py`, or `evaluate.py` may download pretrained `torchvision` weights if they are not already cached.
- `predict.py` falls back to whole-image classification if the Mask R-CNN segmenter cannot be loaded.
- Generated result paths inside old JSON/report files may reflect the machine path used when those files were created. Re-running `evaluate.py` regenerates them for the current workspace.
