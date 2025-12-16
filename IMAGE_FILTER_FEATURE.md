# Image Filter Feature

## Overview
The Image Filter feature allows users to automatically filter images in a folder based on YOLO model detections when opening a folder. This helps users quickly focus on images that contain objects of interest.

## How It Works

### 1. Opening a Folder
When you select **File > Open Folder** (or use Ctrl+U), after choosing a directory:
1. The system scans all images in the selected folder
2. An **Image Filter Dialog** appears with filtering options
3. You can choose to load all images or filter by detections

### 2. Filter Options

#### No Filtering (Default)
- Select "Load all images (no filtering)" to load all images in the folder
- This is the traditional behavior

#### Filter by Detections
- Select "Filter images by detections" to enable filtering
- **Model**: Uses the currently loaded YOLO model
- **Minimum Confidence**: Set a threshold (0.0 - 1.0) for detections
  - Only images with at least one detection above this threshold will be included
  - Default: 0.50 (50%)

### 3. Progress Tracking
During filtering:
- Progress bar shows the scanning progress
- Status shows current image being processed (e.g., "Processing: 50/100 images")
- Matched count shows how many images passed the filter (e.g., "Matched: 35 images")

### 4. Results
- After filtering completes, only images with detections above the threshold are loaded
- A summary message shows how many images matched the filter
- You can cancel the filtering at any time

## Configuration

Filter settings are automatically saved and restored:
- Last used filter mode (enabled/disabled)
- Last used confidence threshold

Settings are stored in `~/.anylabelingrc` under the `image_filter` section:
```yaml
image_filter:
  enabled: false
  min_confidence: 0.5
```

## Requirements

- A YOLO model must be loaded before using the filter feature
- Supported model types: YOLOv5, YOLOv8, YOLOv11
- The application uses the currently loaded model for filtering

## Use Cases

1. **Quality Control**: Filter to show only images with detections for review
2. **Dataset Curation**: Identify images that contain objects of interest
3. **Efficient Labeling**: Skip images with no objects detected
4. **Confidence-Based Filtering**: Find images with high-confidence detections for validation

## Technical Details

- Filtering runs in a background thread to avoid freezing the UI
- Images are processed one at a time for memory efficiency
- The filtering can be cancelled at any time
- Detection results are not saved (only used for filtering)

## Limitations

- Only works with YOLO-based models
- Cannot filter by specific object classes (filters based on any detection)
- Filtering process can take time for large image folders
- Does not cache results (filtering is performed each time a folder is opened)
