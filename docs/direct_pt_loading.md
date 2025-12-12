# Direct .pt File Loading for YOLO Models

## Overview

AnyLabeling now supports direct loading of `.pt` model files for YOLOv5, YOLOv8, and YOLO11 models without requiring manual configuration file creation.

## Features

### 1. Direct Model Loading

You can now load YOLO models by simply selecting the `.pt` file:

1. Click "Load Custom Model" in the auto-labeling widget
2. Select a `.pt` file (e.g., `yolov8n.pt`, `yolo11n.pt`)
3. The system will automatically:
   - Extract class names from the model
   - Detect the model type (v5, v8, or v11)
   - Generate a `config.yaml` file in the same directory
   - Load the model and display it in the UI

### 2. Auto-Generated Configuration

The generated `config.yaml` includes:

```yaml
type: yolov8  # or yolov5, yolov11
name: yolov8n
display_name: Yolov8n
model_path: yolov8n.pt
input_width: 640
input_height: 640
score_threshold: 0.25
nms_threshold: 0.45
confidence_threshold: 0.25
classes:
  - person
  - bicycle
  # ... all detected classes
```

### 3. Live Threshold Controls

When a YOLO model is loaded, threshold controls appear in the UI:

- **Confidence Threshold** (0.0 - 1.0): Minimum confidence for detections
- **Score Threshold** (0.0 - 1.0): Minimum class prediction score
- **NMS Threshold** (0.0 - 1.0): Non-Maximum Suppression overlap threshold

These controls update in real-time - adjust the values and click "Run" to see the effect on detections.

## Requirements

Direct `.pt` file loading requires the ultralytics package:

```bash
pip install ultralytics
```

If ultralytics is not installed, the system will display an error message with installation instructions.

## Supported Model Types

- **YOLOv5**: Detection models from the YOLOv5 repository
- **YOLOv8**: Detection models from Ultralytics YOLOv8
- **YOLO11**: Latest YOLO11 models from Ultralytics

All models use the same ultralytics YOLO API for inference.

## Model Type Detection

The system attempts to detect the model type using:

1. Model metadata and architecture information
2. Filename patterns (e.g., `yolov5n.pt`, `yolo11s.pt`)

If detection fails, the system defaults to YOLOv8, which is compatible with most modern YOLO models.

## Notes

- The generated `config.yaml` file is saved in the same directory as your `.pt` file
- You can manually edit the `config.yaml` after generation if needed
- The model will be added to your custom models list for easy reloading
- Up to 5 custom models are cached; older models are removed automatically
