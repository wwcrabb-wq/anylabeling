# Testing the Image Filter Feature

This guide explains how to test the new Image Filter feature.

## Prerequisites

1. Install AnyLabeling with all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Have a folder of test images ready
3. Have a YOLO model loaded in the application

## Test Scenarios

### 1. Basic Functionality Test

**Test: Open folder with no filtering**
1. Launch AnyLabeling
2. Click **File > Open Folder** (or press Ctrl+U)
3. Select a folder containing images
4. In the Image Filter Dialog, verify:
   - "Load all images (no filtering)" is selected by default
   - Progress section shows "Ready to filter"
5. Click **Apply Filter**
6. Verify all images in the folder are loaded

**Expected Result**: All images load normally, just like before the feature was added.

---

### 2. Filter with Model Loaded

**Test: Filter images with detections**
1. Launch AnyLabeling
2. Load a YOLO model (**Tools > Auto Labeling > Load Model**)
3. Click **File > Open Folder**
4. Select a folder with test images (mix of images with/without objects)
5. In the Image Filter Dialog:
   - Select "Filter images by detections"
   - Verify the model combo shows the loaded model
   - Set confidence threshold to 0.50
   - Click **Apply Filter**
6. Watch the progress bar advance
7. Note the matched count
8. Click OK when filtering completes

**Expected Result**: 
- Only images with detections above 0.50 confidence are loaded
- Progress bar shows real-time progress
- Final message shows count of matched images

---

### 3. No Model Loaded Test

**Test: Attempt filtering without a model**
1. Launch AnyLabeling (don't load a model)
2. Click **File > Open Folder**
3. Select a folder
4. In the Image Filter Dialog:
   - Select "Filter images by detections"
   - Click **Apply Filter**

**Expected Result**: Warning message appears: "Please load a model before filtering images."

---

### 4. Confidence Threshold Test

**Test: Different confidence thresholds**
1. Load a YOLO model
2. Open the same folder multiple times with different thresholds:
   - 0.10 (low) - should match more images
   - 0.50 (medium) - should match fewer images
   - 0.90 (high) - should match only very confident detections

**Expected Result**: 
- Lower thresholds include more images
- Higher thresholds include fewer images
- Matched count changes appropriately

---

### 5. Settings Persistence Test

**Test: Settings are saved and restored**
1. Open a folder with filtering enabled
2. Set confidence to 0.75
3. Click **Apply Filter**
4. Close AnyLabeling
5. Restart AnyLabeling
6. Open a folder again
7. Check the Image Filter Dialog

**Expected Result**: 
- Filter mode should still be "Filter images by detections"
- Confidence threshold should be 0.75

---

### 6. Cancel Operation Test

**Test: Cancel filtering mid-process**
1. Load a YOLO model
2. Open a folder with many images (50+ images)
3. Select "Filter images by detections"
4. Click **Apply Filter**
5. While processing, click **Cancel**

**Expected Result**: 
- Filtering stops
- Dialog closes
- No images are loaded
- Application remains responsive

---

### 7. Large Folder Test

**Test: Performance with large datasets**
1. Load a YOLO model
2. Open a folder with 100+ images
3. Enable filtering
4. Monitor:
   - UI responsiveness
   - Progress updates
   - Memory usage

**Expected Result**: 
- UI remains responsive (not frozen)
- Progress bar updates smoothly
- Memory doesn't spike excessively

---

### 8. Empty Results Test

**Test: No images match filter**
1. Load a YOLO model
2. Open a folder with images that have no detections
3. Set confidence threshold to 0.50
4. Apply filter

**Expected Result**: 
- Message shows "Found 0 images with detections"
- No images are loaded
- Application remains stable

---

## Configuration Verification

Check `~/.anylabelingrc` after running tests:

```yaml
# Should contain:
image_filter:
  enabled: true  # or false, based on last selection
  min_confidence: 0.5  # or your last used value
```

## Known Limitations

1. **Model Type**: Only works with YOLO models (YOLOv5, YOLOv8, YOLOv11)
2. **No Caching**: Filtering is performed fresh each time a folder is opened
3. **No Class Filtering**: Filters based on any detection, not specific classes
4. **Sequential Processing**: Images are processed one at a time

## Troubleshooting

### Issue: Dialog doesn't appear
**Check**: Make sure you selected a valid folder with images

### Issue: Filtering is very slow
**Possible causes**:
- Large images (try with smaller images)
- Complex model (try with a lighter model)
- Many images (this is expected)

### Issue: No images match filter
**Possible causes**:
- Confidence threshold is too high
- Images don't contain detectable objects
- Model is not appropriate for the image content

### Issue: Settings not persisting
**Check**: `~/.anylabelingrc` file permissions and content

## Manual Inspection Points

When testing, verify:

1. **UI Elements**:
   - [ ] Dialog has proper title and layout
   - [ ] Radio buttons work correctly
   - [ ] Slider updates value label
   - [ ] Progress bar animates smoothly
   - [ ] Buttons are properly enabled/disabled

2. **Functionality**:
   - [ ] Filtering works with different models
   - [ ] Cancel button stops processing
   - [ ] Settings save and load correctly
   - [ ] Error handling works (no model, no images, etc.)

3. **Performance**:
   - [ ] UI doesn't freeze during filtering
   - [ ] Memory usage is reasonable
   - [ ] Progress updates in real-time

4. **Edge Cases**:
   - [ ] Empty folder
   - [ ] Folder with one image
   - [ ] Very large folder (500+ images)
   - [ ] Corrupted images (should be skipped gracefully)

## Reporting Issues

If you find issues, please report:
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Model type and version used
- Number of images in test folder
- AnyLabeling version
