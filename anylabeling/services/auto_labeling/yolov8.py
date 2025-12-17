import functools
import logging
import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class YOLOv8(Model):
    """Object detection model using YOLOv8"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "score_threshold",
            "nms_threshold",
            "confidence_threshold",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )

        self.net = None
        self.ultralytics_model = None
        self.use_ultralytics = False

        # Try loading with ultralytics if model is .pt
        if model_abs_path.endswith(".pt"):
            try:
                # Import here to avoid dependency errors when ultralytics is not installed
                import torch
                from ultralytics import YOLO

                # Patch torch.load to use weights_only=False for trusted user-selected models
                # This is safe because users explicitly select these model files
                original_load = torch.load

                @functools.wraps(original_load)
                def _patched_load(*args, **kwargs):
                    if "weights_only" not in kwargs:
                        kwargs["weights_only"] = False
                    return original_load(*args, **kwargs)

                try:
                    torch.load = _patched_load
                    self.ultralytics_model = YOLO(model_abs_path)
                finally:
                    torch.load = original_load
                self.use_ultralytics = True
                logging.info("Loaded YOLOv8 model using ultralytics")
            except ImportError:
                logging.warning(
                    "Ultralytics not available. To use .pt models, install: pip install ultralytics"
                )
                self.on_message(
                    QCoreApplication.translate(
                        "Model",
                        "Ultralytics not installed. Install with: pip install ultralytics",
                    )
                )
            except Exception as e:
                logging.warning(f"Failed to load .pt with ultralytics: {e}")

        # Fallback to cv2.dnn if ultralytics failed or not a .pt file
        if not self.use_ultralytics:
            self.net = cv2.dnn.readNet(model_abs_path)
            if __preferred_device__ == "GPU":
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.classes = self.config["classes"]

    def pre_process(self, input_image, net):
        """
        Pre-process the input image before feeding it to the network.
        """
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(
            input_image,
            1 / 255,
            (self.config["input_width"], self.config["input_height"]),
            [0, 0, 0],
            1,
            crop=False,
        )

        # Sets the input to the network.
        net.setInput(blob)

        # Runs the forward with blob.
        outputs = net.forward()
        outputs = np.array([cv2.transpose(outputs[0])])

        return outputs

    def post_process(self, input_image, outputs):
        """
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        """
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []

        # Rows.
        rows = outputs.shape[1]

        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.config["input_width"]
        y_factor = image_height / self.config["input_height"]

        # Read thresholds from config at prediction time for live updates
        confidence_threshold = self.config.get("confidence_threshold", 0.25)
        nms_threshold = self.config.get("nms_threshold", 0.45)

        # Iterate through 8400 rows.
        for r in range(rows):
            row = outputs[0][r]
            classes_scores = row[4:]

            # Get the index of max class score and confidence.
            _, confidence, _, (_, class_id) = cv2.minMaxLoc(classes_scores)

            # Discard confidence lower than threshold
            if confidence >= confidence_threshold:
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            confidence_threshold,
            nms_threshold,
        )

        output_boxes = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            label = self.classes[class_ids[i]]
            score = confidences[i]

            output_box = {
                "x1": left,
                "y1": top,
                "x2": left + width,
                "y2": top + height,
                "label": label,
                "score": score,
            }

            output_boxes.append(output_box)

        return output_boxes

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        if self.use_ultralytics and self.ultralytics_model:
            boxes = self._predict_ultralytics(image)
        else:
            detections = self.pre_process(image, self.net)
            boxes = self.post_process(image, detections)

        shapes = []
        for box in boxes:
            shape = Shape(label=box["label"], shape_type="rectangle", flags={})
            shape.add_point(QtCore.QPointF(box["x1"], box["y1"]))
            shape.add_point(QtCore.QPointF(box["x2"], box["y2"]))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def predict_shapes_batch(self, images, image_paths=None):
        """
        Predict shapes from multiple images in a batch for improved performance.

        Args:
            images: List of images (numpy arrays or QImage objects)
            image_paths: Optional list of image file paths

        Returns:
            List of AutoLabelingResult objects, one per image
        """
        if not images:
            return []

        if image_paths is None:
            image_paths = [None] * len(images)

        # Convert all images to cv2 format
        cv_images = []
        for img, img_path in zip(images, image_paths):
            if img is None:
                cv_images.append(None)
                continue
            try:
                cv_img = qt_img_to_rgb_cv_img(img, img_path)
                cv_images.append(cv_img)
            except Exception as e:
                logging.warning(f"Could not convert image: {e}")
                cv_images.append(None)

        # Process batch
        if self.use_ultralytics and self.ultralytics_model:
            # Ultralytics can process batches natively
            all_boxes = self._predict_ultralytics_batch(cv_images)
        else:
            # For cv2.dnn, process individually (batch support limited)
            all_boxes = []
            for cv_img in cv_images:
                if cv_img is not None:
                    detections = self.pre_process(cv_img, self.net)
                    boxes = self.post_process(cv_img, detections)
                    all_boxes.append(boxes)
                else:
                    all_boxes.append([])

        # Convert to shapes
        results = []
        for boxes in all_boxes:
            shapes = []
            for box in boxes:
                shape = Shape(label=box["label"], shape_type="rectangle", flags={})
                shape.add_point(QtCore.QPointF(box["x1"], box["y1"]))
                shape.add_point(QtCore.QPointF(box["x2"], box["y2"]))
                shapes.append(shape)
            results.append(AutoLabelingResult(shapes, replace=True))

        return results

    def _predict_ultralytics(self, image):
        """
        Run prediction using ultralytics model
        """
        try:
            # Read thresholds from config at prediction time for live updates
            conf_threshold = self.config.get("confidence_threshold", 0.25)
            nms_threshold = self.config.get("nms_threshold", 0.45)

            # Run prediction
            results = self.ultralytics_model.predict(
                source=image, conf=conf_threshold, iou=nms_threshold, verbose=False
            )

            boxes = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())

                        # Get label from classes or use default
                        if cls_id < len(self.classes):
                            label = self.classes[cls_id]
                        else:
                            label = f"AUTOLABEL_OBJECT_{cls_id}"

                        boxes.append(
                            {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "label": label,
                                "score": conf,
                            }
                        )

            return boxes
        except Exception as e:
            logging.warning(f"Ultralytics prediction failed: {e}")
            return []

    def _predict_ultralytics_batch(self, images):
        """
        Run batch prediction using ultralytics model for improved performance.

        Args:
            images: List of cv2 images (numpy arrays)

        Returns:
            List of lists of boxes, one per image
        """
        try:
            # Filter out None images and track indices
            valid_images = []
            valid_indices = []
            for idx, img in enumerate(images):
                if img is not None:
                    valid_images.append(img)
                    valid_indices.append(idx)

            if not valid_images:
                return [[] for _ in images]

            # Read thresholds from config at prediction time for live updates
            conf_threshold = self.config.get("confidence_threshold", 0.25)
            nms_threshold = self.config.get("nms_threshold", 0.45)

            # Run batch prediction
            results = self.ultralytics_model.predict(
                source=valid_images,
                conf=conf_threshold,
                iou=nms_threshold,
                verbose=False,
            )

            # Initialize results for all images
            all_boxes = [[] for _ in images]

            # Process results for valid images
            for result_idx, result in enumerate(results):
                boxes = []
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())

                        # Get label from classes or use default
                        if cls_id < len(self.classes):
                            label = self.classes[cls_id]
                        else:
                            label = f"AUTOLABEL_OBJECT_{cls_id}"

                        boxes.append(
                            {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "label": label,
                                "score": conf,
                            }
                        )

                # Map back to original index
                original_idx = valid_indices[result_idx]
                all_boxes[original_idx] = boxes

            return all_boxes
        except Exception as e:
            logging.warning(f"Ultralytics batch prediction failed: {e}")
            return [[] for _ in images]

    def unload(self):
        if self.net is not None:
            del self.net
        if self.ultralytics_model is not None:
            del self.ultralytics_model

    def _on_config_param_changed(self, key, value):
        """
        Hook for config parameter changes.
        Can be extended in the future for specific parameter handling.
        """
        pass
