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


class YOLOv5(Model):
    """Object detection model using YOLOv5"""

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
                    "Model", "Could not download or initialize YOLOv5 model."
                )
            )

        self.net = None
        self.ultralytics_model = None
        self.use_ultralytics = False

        # Try loading with ultralytics if model is .pt
        if model_abs_path.endswith(".pt"):
            try:
                import torch
                from ultralytics import YOLO

                # Register ultralytics classes as safe globals for PyTorch 2.6+ compatibility
                if hasattr(torch.serialization, "add_safe_globals"):
                    try:
                        from ultralytics.nn.tasks import (
                            DetectionModel,
                            SegmentationModel,
                            ClassificationModel,
                            PoseModel,
                        )

                        torch.serialization.add_safe_globals(
                            [
                                DetectionModel,
                                SegmentationModel,
                                ClassificationModel,
                                PoseModel,
                            ]
                        )
                    except (ImportError, AttributeError):
                        pass  # Older ultralytics version or classes not available

                self.ultralytics_model = YOLO(model_abs_path)
                self.use_ultralytics = True
                logging.info("Loaded YOLOv5 model using ultralytics")
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

        # Runs the forward pass to get output of the output layers.
        output_layers = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers)

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
        rows = outputs[0].shape[1]

        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.config["input_width"]
        y_factor = image_height / self.config["input_height"]

        # Read thresholds from config at prediction time for live updates
        confidence_threshold = self.config.get("confidence_threshold", 0.25)
        score_threshold = self.config.get("score_threshold", 0.25)
        nms_threshold = self.config.get("nms_threshold", 0.45)

        # Iterate through 25200 detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]

            # Discard bad detections and continue.
            if confidence >= confidence_threshold:
                classes_scores = row[5:]

                # Get the index of max class score.
                class_id = np.argmax(classes_scores)

                #  Continue if the class score is above threshold.
                if classes_scores[class_id] > score_threshold:
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
