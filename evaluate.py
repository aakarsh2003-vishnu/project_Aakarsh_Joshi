# evaluate.py
# Runs the final testing, segmentation demo, and report generation.

import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from dataset import VehicleDataset
from model import VehicleClassifier
from predict import predict_vehicles
from train import evaluate_model, make_stratified_split


def _load_trained_model():
    model = VehicleClassifier(pretrained=False)
    state = torch.load(config.WEIGHTS_PATH, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


def _sample_demo_images():
    paths = []
    folder_names = {
        "car": "Car",
        "bus": "Bus",
        "truck": "Truck",
        "motorcycle": "motorcycle",
    }
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label in config.class_names:
        folder = folder_names[label]
        folder_path = os.path.join(config.DATA_DIR, folder)
        for fname in sorted(os.listdir(folder_path)):
            if os.path.splitext(fname)[1].lower() in valid_ext:
                paths.append(os.path.join(folder_path, fname))
                break
    return paths


def _read_training_history():
    if not os.path.exists(config.TRAIN_HISTORY_PATH):
        return None
    with open(config.TRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_report(results):
    lines = []
    lines.append("# Vehicle Segmentation and Classification Report")
    lines.append("")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Proposal Alignment")
    lines.append("")
    lines.append("The proposal asked for a two-stage computer vision system for street images:")
    lines.append("1. Segment or detect vehicle regions from the image background.")
    lines.append("2. Classify each vehicle crop as car, bus, truck, or motorcycle.")
    lines.append("")
    lines.append("This implementation follows that plan. Mask R-CNN is used for the segmentation/detection stage when available, and a fine-tuned ResNet-18 classifier is used for vehicle type classification.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"Total images: {results['dataset']['total_images']}")
    for name, count in results["dataset"]["class_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append(f"Training images: {results['classifier']['train_samples']}")
    lines.append(f"Testing images: {results['classifier']['test_samples']}")
    lines.append("")
    lines.append("## Training Setup")
    lines.append("")
    lines.append("- Backbone: ResNet-18")
    lines.append("- Final layer: custom 4-class classifier")
    lines.append(f"- Epochs: {results['classifier']['epochs']}")
    lines.append(f"- Batch size: {results['classifier']['batch_size']}")
    lines.append(f"- Learning rate: {results['classifier']['learning_rate']}")
    lines.append("- Loss: CrossEntropyLoss")
    lines.append("- Optimizer: Adam")
    lines.append("")
    lines.append("## Classifier Test Results")
    lines.append("")
    metrics = results["classifier"]["test_metrics"]
    lines.append(f"Test accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Test loss: {metrics['loss']:.4f}")
    lines.append("")
    lines.append("Per-class accuracy:")
    for name, acc in metrics["per_class_accuracy"].items():
        lines.append(f"- {name}: {acc:.4f}")
    lines.append("")
    lines.append("Confusion matrix rows are true classes and columns are predicted classes in this order: car, bus, truck, motorcycle.")
    lines.append("")
    lines.append("```")
    for row in metrics["confusion_matrix"]:
        lines.append(str(row))
    lines.append("```")
    lines.append("")
    lines.append("## Segmentation and Full Pipeline Test")
    lines.append("")
    lines.append("The full pipeline was run on one sample image from each class. The generated output images show detected vehicle regions, segmentation masks where available, bounding boxes, and final predicted labels.")
    lines.append("")
    for item in results["pipeline_results"]:
        lines.append(f"- Image: {item['image_path']}")
        lines.append(f"  Detections: {item['detections']}")
        if "annotated_image" in item:
            lines.append(f"  Output: {item['annotated_image']}")
        for vehicle in item["vehicles"]:
            lines.append(
                f"  Vehicle {vehicle['crop_index']}: detector={vehicle['det_label']} "
                f"({vehicle['det_score']}), classifier={vehicle['pred_label']} "
                f"({vehicle['pred_confidence']})"
            )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("The project now contains a complete runnable prototype matching the proposal: dataset loading, classifier training, held-out testing, segmentation/detection, crop classification, saved visual outputs, and a reproducible report.")

    with open(config.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    dataset = VehicleDataset(root_dir=config.DATA_DIR, augment=False)
    train_indices, test_indices = make_stratified_split(dataset)
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=0,
    )

    class_counts = {name: 0 for name in config.class_names}
    for _, label in dataset.samples:
        class_counts[config.class_names[label]] += 1

    model = _load_trained_model()
    metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss())

    training_history = _read_training_history()
    classifier_section = {
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "epochs": config.epochs,
        "batch_size": config.batchsize,
        "learning_rate": config.lr,
        "test_metrics": metrics,
    }
    if training_history and "history" in training_history:
        classifier_section["history"] = training_history["history"]

    pipeline_results = predict_vehicles(_sample_demo_images(), save_outputs=True)

    results = {
        "dataset": {
            "total_images": len(dataset),
            "class_counts": class_counts,
        },
        "classifier": classifier_section,
        "pipeline_results": pipeline_results,
    }

    with open(config.EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_report(results)

    print(f"Evaluation saved to {config.EVAL_RESULTS_PATH} - evaluate.py:174")
    print(f"Report saved to {config.REPORT_PATH} - evaluate.py:175")
    print(f"Visual outputs saved to {config.VISUALS_DIR} - evaluate.py:176")
    print(f"Test accuracy: {metrics['accuracy']:.4f} - evaluate.py:177")


if __name__ == "__main__":
    main()
