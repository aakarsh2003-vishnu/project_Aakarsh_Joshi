# Vehicle Segmentation and Classification Report

Generated on: 2026-05-02 20:11:41

## Proposal Alignment

The proposal asked for a two-stage computer vision system for street images:
1. Segment or detect vehicle regions from the image background.
2. Classify each vehicle crop as car, bus, truck, or motorcycle.

This implementation follows that plan. Mask R-CNN is used for the segmentation/detection stage when available, and a fine-tuned ResNet-18 classifier is used for vehicle type classification.

## Dataset

Total images: 400
- car: 100
- bus: 100
- truck: 100
- motorcycle: 100

Training images: 320
Testing images: 80

## Training Setup

- Backbone: ResNet-18
- Final layer: custom 4-class classifier
- Epochs: 10
- Batch size: 16
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Classifier Test Results

Test accuracy: 0.9250
Test loss: 0.2197

Per-class accuracy:
- car: 0.9500
- bus: 0.9000
- truck: 0.8500
- motorcycle: 1.0000

Confusion matrix rows are true classes and columns are predicted classes in this order: car, bus, truck, motorcycle.

```
[19, 0, 1, 0]
[1, 18, 1, 0]
[1, 2, 17, 0]
[0, 0, 0, 20]
```

## Segmentation and Full Pipeline Test

The full pipeline was run on one sample image from each class. The generated output images show detected vehicle regions, segmentation masks where available, bounding boxes, and final predicted labels.

- Image: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\data\Car\Image_1.jpg
  Detections: 2
  Output: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\results\segmentation_outputs\Car_Image_1_segmented.jpg
  Vehicle 0: detector=car (0.8636), classifier=car (0.9994)
  Vehicle 1: detector=motorcycle (0.5971), classifier=car (0.9996)
- Image: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\data\Bus\Image_1.jpg
  Detections: 1
  Output: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\results\segmentation_outputs\Bus_Image_1_segmented.jpg
  Vehicle 0: detector=bus (0.9968), classifier=bus (0.9989)
- Image: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\data\Truck\Image_1.jpg
  Detections: 1
  Output: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\results\segmentation_outputs\Truck_Image_1_segmented.jpg
  Vehicle 0: detector=truck (0.9472), classifier=truck (0.9999)
- Image: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\data\motorcycle\Image_1.jpeg
  Detections: 1
  Output: C:\Users\AAKARSH JOSHI\OneDrive\Desktop\project_student_name\project_Aakarsh_Joshi\results\segmentation_outputs\motorcycle_Image_1_segmented.jpg
  Vehicle 0: detector=motorcycle (0.9983), classifier=motorcycle (0.9989)

## Conclusion

The project now contains a complete runnable prototype matching the proposal: dataset loading, classifier training, held-out testing, segmentation/detection, crop classification, saved visual outputs, and a reproducible report.