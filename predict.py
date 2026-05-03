# predict.py
# Two stage pipeline:
# 1. Segment/detect vehicles with a COCO Mask R-CNN model.
# 2. Classify each vehicle crop with the trained VehicleClassifier.

import json
import os
from typing import List, Union

import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms

import config
from model import VehicleClassifier


_segmenter = None
_classifier = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_segmenter():
    """Load Mask R-CNN for COCO vehicle masks."""
    global _segmenter
    if _segmenter is None:
        print("[predict] Loading Mask R-CNN segmenter...")
        try:
            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            _segmenter = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
            _segmenter.to(_get_device()).eval()
            print("[predict] Segmenter ready.")
        except Exception as exc:
            print(f"[predict] WARNING: segmenter unavailable: {exc}")
            print("[predict] Falling back to whole-image classification.")
            _segmenter = False
    return _segmenter


def _load_classifier():
    """Load the trained classifier checkpoint."""
    global _classifier
    if _classifier is None:
        model = VehicleClassifier(pretrained=False)
        if os.path.exists(config.WEIGHTS_PATH):
            state = torch.load(config.WEIGHTS_PATH, map_location=_get_device())
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            print(f"[predict] Loaded classifier weights from {config.WEIGHTS_PATH}")
        else:
            print("[predict] WARNING: trained classifier weights were not found.")
        model.to(_get_device()).eval()
        _classifier = model
    return _classifier


_COCO_VEHICLE_IDS = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

_TO_TENSOR = transforms.ToTensor()
_CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((config.resize_y, config.resize_x)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _segment_vehicles(pil_image):
    """Return crops and metadata for detected vehicle masks."""
    segmenter = _load_segmenter()
    if segmenter is False:
        width, height = pil_image.size
        return [{
            "crop": pil_image,
            "det_label": "whole_image",
            "score": 1.0,
            "box": [0, 0, width, height],
            "mask": None,
        }]

    img_tensor = _TO_TENSOR(pil_image).to(_get_device())
    with torch.no_grad():
        pred = segmenter([img_tensor])[0]

    results = []
    for box, score, cid, mask in zip(
        pred["boxes"].cpu(),
        pred["scores"].cpu(),
        pred["labels"].cpu(),
        pred["masks"].cpu(),
    ):
        cid = int(cid.item())
        score_value = float(score.item())
        if cid not in _COCO_VEHICLE_IDS:
            continue
        if score_value < config.SEG_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        if (x2 - x1) * (y2 - y1) < config.MIN_BOX_AREA:
            continue

        crop = pil_image.crop((x1, y1, x2, y2))
        results.append({
            "crop": crop,
            "det_label": _COCO_VEHICLE_IDS[cid],
            "score": score_value,
            "box": [x1, y1, x2, y2],
            "mask": mask[0] > 0.5,
        })

    return results


def _classify_crops(crops: List[Image.Image]):
    if not crops:
        return []

    classifier = _load_classifier()
    batch = torch.stack([_CLASSIFIER_TRANSFORM(c) for c in crops]).to(_get_device())
    with torch.no_grad():
        logits = classifier(batch)
        probs = torch.softmax(logits, dim=1)

    output = []
    for row in probs.cpu():
        idx = int(row.argmax().item())
        output.append({
            "label": config.class_names[idx],
            "confidence": float(row[idx].item()),
        })
    return output


def _safe_name(path):
    parent = os.path.basename(os.path.dirname(path))
    base = parent + "_" + os.path.splitext(os.path.basename(path))[0]
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base)


def save_visual_outputs(image_path, detections, vehicles, output_dir=None):
    """Save an annotated image and individual vehicle crops."""
    output_dir = output_dir or config.VISUALS_DIR
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_image = ImageDraw.Draw(image)

    stem = _safe_name(image_path)
    crop_paths = []

    for idx, (det, vehicle) in enumerate(zip(detections, vehicles)):
        x1, y1, x2, y2 = det["box"]
        label = f"{vehicle['pred_label']} {vehicle['pred_confidence']:.2f}"

        if det["mask"] is not None:
            mask_img = Image.fromarray((det["mask"].numpy() * 120).astype("uint8"), mode="L")
            color = Image.new("RGBA", image.size, (40, 180, 120, 0))
            color.putalpha(mask_img)
            overlay = Image.alpha_composite(overlay, color)

        draw_image.rectangle((x1, y1, x2, y2), outline=(255, 220, 0), width=4)
        draw_image.text((x1 + 4, max(0, y1 - 16)), label, fill=(255, 220, 0))

        crop_path = os.path.join(output_dir, f"{stem}_crop_{idx}.jpg")
        det["crop"].save(crop_path)
        crop_paths.append(crop_path)

    final = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    annotated_path = os.path.join(output_dir, f"{stem}_segmented.jpg")
    final.save(annotated_path)
    return annotated_path, crop_paths


def predict_vehicles(img_paths: Union[str, List[str]], save_outputs=False):
    """Detect/segment and classify vehicles in one or more images."""
    if isinstance(img_paths, str):
        img_paths = [img_paths]

    all_results = []

    for path in img_paths:
        pil_img = Image.open(path).convert("RGB")
        detections = _segment_vehicles(pil_img)
        crops = [item["crop"] for item in detections]
        predictions = _classify_crops(crops)

        vehicles = []
        for i, (det, pred) in enumerate(zip(detections, predictions)):
            vehicles.append({
                "crop_index": i,
                "det_label": det["det_label"],
                "det_score": round(det["score"], 4),
                "box": det["box"],
                "pred_label": pred["label"],
                "pred_confidence": round(pred["confidence"], 4),
            })

        result = {
            "image_path": path,
            "detections": len(vehicles),
            "vehicles": vehicles,
        }

        if save_outputs:
            annotated, crops_saved = save_visual_outputs(path, detections, vehicles)
            result["annotated_image"] = annotated
            result["crop_paths"] = crops_saved

        all_results.append(result)

    return all_results


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if a != "--save"]
    save_outputs = "--save" in sys.argv[1:]

    if args:
        paths = args
    else:
        paths = []
        for class_name in ["Car", "Bus", "Truck", "motorcycle"]:
            sample = os.path.join(config.DATA_DIR, class_name, "Image_1.jpg")
            if os.path.exists(sample):
                paths.append(sample)

    results = predict_vehicles(paths, save_outputs=save_outputs)
    print(json.dumps(results, indent=2))
