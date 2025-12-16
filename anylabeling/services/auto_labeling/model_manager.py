import os
import copy
import time
import shutil
import pathlib
import logging
import tempfile
import zipfile
import importlib.resources as pkg_resources
from threading import Lock
import urllib.request

import yaml
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtCore import QCoreApplication

from anylabeling.configs import auto_labeling as auto_labeling_configs
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.utils import GenericWorker

from anylabeling.config import get_config, save_config

import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Prevent issue when downloading models behind a proxy


class ModelManager(QObject):
    """Model manager"""

    MAX_NUM_CUSTOM_MODELS = 5

    model_configs_changed = pyqtSignal(list)
    new_model_status = pyqtSignal(str)
    model_loaded = pyqtSignal(dict)
    new_auto_labeling_result = pyqtSignal(AutoLabelingResult)
    auto_segmentation_model_selected = pyqtSignal()
    auto_segmentation_model_unselected = pyqtSignal()
    prediction_started = pyqtSignal()
    prediction_finished = pyqtSignal()
    request_next_files_requested = pyqtSignal()
    output_modes_changed = pyqtSignal(dict, str)

    def __init__(self):
        super().__init__()
        self.model_configs = []

        self.loaded_model_config = None
        self.loaded_model_config_lock = Lock()

        self.model_download_worker = None
        self.model_download_thread = None
        self.model_execution_thread = None
        self.model_execution_worker = None
        self.model_execution_thread_lock = Lock()

        self.load_model_configs()

    def load_model_configs(self):
        """Load model configs"""
        # Load list of default models
        with pkg_resources.open_text(auto_labeling_configs, "models.yaml") as f:
            model_list = yaml.safe_load(f)
            for model in model_list:
                model["is_custom_model"] = False

            # Check downloaded
            for model in model_list:
                home_dir = os.path.expanduser("~")
                model_download_path = os.path.join(
                    home_dir, "anylabeling_data", "models", model["name"]
                )
                pathlib.Path(model_download_path).mkdir(parents=True, exist_ok=True)
                config_file = os.path.join(model_download_path, "config.yaml")
                model["config_file"] = config_file

                # Initialize model config if needed
                if not os.path.isfile(config_file):
                    model["has_downloaded"] = False
                    with open(config_file, "w") as f:
                        yaml.dump(model, f)

        # Load list of custom models
        custom_models = get_config().get("custom_models", [])
        for custom_model in custom_models:
            custom_model["is_custom_model"] = True
            custom_model["has_downloaded"] = True

        # Remove invalid/not found custom models
        custom_models = [
            custom_model
            for custom_model in custom_models
            if os.path.isfile(custom_model.get("config_file", ""))
        ]
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        model_list += custom_models

        # Load model configs
        model_configs = []
        for model in model_list:
            model_config = copy.deepcopy(model)
            config_file = model.get("config_file", None)
            if config_file:
                with open(config_file, "r") as f:
                    model_config = yaml.safe_load(f)
                    model_config["config_file"] = os.path.normpath(
                        os.path.abspath(config_file)
                    )
                    model_config["is_custom_model"] = model.get(
                        "is_custom_model", False
                    )
            model_configs.append(model_config)

        # Sort by last used
        for i, model_config in enumerate(model_configs):
            # Keep order for integrated models
            if not model_config.get("is_custom_model", False):
                model_config["last_used"] = -i
            else:
                model_config["last_used"] = model_config.get("last_used", time.time())
        model_configs.sort(key=lambda x: x.get("last_used", 0), reverse=True)

        self.model_configs = model_configs
        self.model_configs_changed.emit(model_configs)

    def get_model_configs(self):
        """Return model infos"""
        return self.model_configs

    def set_output_mode(self, mode):
        """Set output mode"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.loaded_model_config["model"].set_output_mode(mode)

    @pyqtSlot(str, object, bool)
    def set_loaded_model_param(self, key, value, persist=False):
        """
        Update a parameter in the loaded model's config.

        Args:
            key: The config parameter key to update
            value: The new value for the parameter
            persist: If True, persist the change to the config file
        """
        with self.loaded_model_config_lock:
            if self.loaded_model_config and self.loaded_model_config.get("model"):
                try:
                    self.loaded_model_config["model"].set_config_param(
                        key, value, persist
                    )
                    # Update the model_config dict as well
                    self.loaded_model_config[key] = value
                    # Emit signal to notify UI of changes
                    self.model_configs_changed.emit(self.model_configs)
                except Exception as e:
                    logging.warning(f"Failed to update model parameter {key}: {e}")

    @pyqtSlot()
    def on_model_download_finished(self):
        """Handle model download thread finished"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.new_model_status.emit(self.tr("Model loaded. Ready for labeling."))
            self.model_loaded.emit(self.loaded_model_config)
            self.output_modes_changed.emit(
                self.loaded_model_config["model"].Meta.output_modes,
                self.loaded_model_config["model"].Meta.default_output_mode,
            )
        else:
            self.model_loaded.emit({})

    def load_custom_model(self, config_file):
        """Run custom model loading in a thread"""
        config_file = os.path.normpath(os.path.abspath(config_file))
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
            print("Another model is being loaded. Please wait for it to finish.")
            return

        # Check config file path
        if not config_file or not os.path.isfile(config_file):
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid path.")
            )
            return

        # If it's a .pt file, auto-generate config
        if config_file.endswith(".pt"):
            try:
                config_file = self._generate_config_from_pt(config_file)
                if not config_file:
                    return
            except Exception as e:
                self.new_model_status.emit(
                    self.tr(f"Error generating config from .pt file: {str(e)}")
                )
                return

        # Check config file content
        model_config = {}
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
            model_config["config_file"] = os.path.abspath(config_file)
        if not model_config:
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid config file.")
            )
            return
        if (
            "type" not in model_config
            or "display_name" not in model_config
            or "name" not in model_config
            or model_config["type"]
            not in ["segment_anything", "yolov5", "yolov8", "yolov11"]
        ):
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid config file format.")
            )
            return

        # Add or replace custom model
        custom_models = get_config().get("custom_models", [])
        matched_index = None
        for i, model in enumerate(custom_models):
            if os.path.normpath(model["config_file"]) == os.path.normpath(config_file):
                matched_index = i
                break
        if matched_index is not None:
            model_config["last_used"] = time.time()
            custom_models[matched_index] = model_config
        else:
            if len(custom_models) >= self.MAX_NUM_CUSTOM_MODELS:
                custom_models.sort(key=lambda x: x.get("last_used", 0), reverse=True)
                removed_model = custom_models.pop()
                # Remove old model folder
                config_file = removed_model["config_file"]
                if os.path.exists(config_file):
                    try:
                        pathlib.Path(config_file).parent.rmdir()
                    except OSError:
                        pass
            custom_models = [model_config] + custom_models

        # Save config
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        # Reload model configs
        self.load_model_configs()

        # Load model
        self.load_model(model_config["config_file"])

    def _generate_config_from_pt(self, pt_file):
        """
        Auto-generate config.yaml from a .pt model file

        Args:
            pt_file: Path to the .pt model file

        Returns:
            Path to the generated config.yaml file
        """
        try:
            import functools
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

        except ImportError:
            self.new_model_status.emit(
                self.tr("Please install ultralytics: pip install ultralytics")
            )
            logging.error(
                "ultralytics not installed. Cannot auto-generate config from .pt file"
            )
            return None

        try:
            # Load model to extract information
            self.new_model_status.emit(self.tr("Generating config from .pt file..."))

            # Temporarily patch torch.load
            torch.load = _patched_load
            try:
                model = YOLO(pt_file)
            finally:
                torch.load = original_load

            # Extract class names
            class_names = list(model.names.values())

            # Auto-detect model type from model metadata
            # YOLO11 models have specific architecture indicators
            # Default to yolov8 as fallback - yolov8 and yolov11 are compatible via ultralytics
            model_type = "yolov8"

            # Try to detect model type from task or architecture
            if hasattr(model, "task"):
                # Check model architecture or metadata
                model_name = (
                    str(model.ckpt.get("model", "")).lower()
                    if hasattr(model, "ckpt") and model.ckpt is not None
                    else ""
                )
                if "v11" in model_name or "yolo11" in model_name:
                    model_type = "yolov11"
                elif "v5" in model_name or "yolov5" in model_name:
                    model_type = "yolov5"
                elif "v8" in model_name or "yolov8" in model_name:
                    model_type = "yolov8"

            # Additional heuristic: check file name
            pt_basename = os.path.basename(pt_file).lower()
            if "yolo11" in pt_basename or "v11" in pt_basename:
                model_type = "yolov11"
            elif "yolov5" in pt_basename or "v5" in pt_basename:
                model_type = "yolov5"
            elif "yolov8" in pt_basename or "v8" in pt_basename:
                model_type = "yolov8"

            # Create config dictionary
            model_filename = os.path.basename(pt_file)
            model_name = os.path.splitext(model_filename)[0]

            config_dict = {
                "type": model_type,
                "name": model_name,
                "display_name": model_name.replace("_", " ").title(),
                "model_path": model_filename,
                "input_width": 640,
                "input_height": 640,
                "score_threshold": 0.25,
                "nms_threshold": 0.45,
                "confidence_threshold": 0.25,
                "classes": class_names,
            }

            # Generate config file in the same directory as the .pt file
            config_file = os.path.join(os.path.dirname(pt_file), "config.yaml")

            with open(config_file, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)

            self.new_model_status.emit(self.tr(f"Config file generated: {config_file}"))
            logging.info(f"Auto-generated config file: {config_file}")

            return config_file

        except Exception as e:
            logging.error(f"Failed to generate config from .pt file: {e}")
            self.new_model_status.emit(self.tr(f"Error: {str(e)}"))
            return None

    def load_model(self, config_file):
        """Run model loading in a thread"""
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
            print("Another model is being loaded. Please wait for it to finish.")
            return
        if not config_file:
            if self.model_download_worker is not None:
                try:
                    self.model_download_worker.finished.disconnect(
                        self.on_model_download_finished
                    )
                except TypeError:
                    pass
            self.unload_model()
            self.new_model_status.emit(self.tr("No model selected."))
            return

        # Check and get model id
        model_id = None
        for i, model_config in enumerate(self.model_configs):
            if model_config["config_file"] == config_file:
                model_id = i
                break
        if model_id is None:
            self.new_model_status.emit(
                self.tr("Error in loading model: Invalid model name.")
            )
            return

        self.model_download_thread = QThread()
        self.new_model_status.emit(
            self.tr("Loading model: {model_name}. Please wait...").format(
                model_name=self.model_configs[model_id]["display_name"]
            )
        )
        self.model_download_worker = GenericWorker(self._load_model, model_id)
        self.model_download_worker.finished.connect(self.on_model_download_finished)
        self.model_download_worker.finished.connect(self.model_download_thread.quit)
        self.model_download_worker.moveToThread(self.model_download_thread)
        self.model_download_thread.started.connect(self.model_download_worker.run)
        self.model_download_thread.start()

    def _download_and_extract_model(self, model_config):
        """Download and extract a model from model config"""
        config_file = model_config["config_file"]
        # Check if model is already downloaded
        if not os.path.exists(config_file):
            raise ValueError(self.tr("Error in loading config file."))
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        if model_config.get("has_downloaded", False):
            return

        # Download model
        download_url = model_config.get("download_url", None)
        if not download_url:
            raise ValueError(self.tr("Missing download_url in config file."))
        tmp_dir = tempfile.mkdtemp()
        zip_model_path = os.path.join(tmp_dir, "model.zip")

        # Download url
        ellipsis_download_url = download_url
        if len(download_url) > 40:
            ellipsis_download_url = download_url[:20] + "..." + download_url[-20:]
        logging.info("Downloading %s to %s", ellipsis_download_url, zip_model_path)
        try:
            # Download and show progress
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                self.new_model_status.emit(
                    QCoreApplication.translate(
                        "Model", "Downloading {download_url}: {percent}%"
                    ).format(download_url=ellipsis_download_url, percent=percent)
                )

            urllib.request.urlretrieve(
                download_url, zip_model_path, reporthook=_progress
            )
        except Exception as e:  # noqa
            print(f"Could not download {download_url}: {e}")
            self.new_model_status.emit(f"Could not download {download_url}")
            return None

        # Extract model
        tmp_extract_dir = os.path.join(tmp_dir, "extract")
        extract_dir = os.path.dirname(config_file)
        with zipfile.ZipFile(zip_model_path, "r") as zip_ref:
            zip_ref.extractall(tmp_extract_dir)

        # Find model folder (containing config.yaml)
        model_folder = None
        for root, _, files in os.walk(tmp_extract_dir):
            if "config.yaml" in files:
                model_folder = root
                break
        if model_folder is None:
            raise ValueError(self.tr("Could not find config.yaml in zip file."))

        # Move model folder to correct location
        shutil.rmtree(extract_dir)
        shutil.move(model_folder, extract_dir)

        # Clean up
        shutil.rmtree(tmp_dir)

        # Update config file
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        model_config["has_downloaded"] = True
        model_config["config_file"] = config_file
        with open(config_file, "w") as f:
            yaml.dump(model_config, f)

        return model_config

    def _load_model(self, model_id):
        """Load and return model info"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None
            self.auto_segmentation_model_unselected.emit()

        model_config = copy.deepcopy(self.model_configs[model_id])

        # Download and extract model
        if not model_config.get("has_downloaded", True):
            model_config = self._download_and_extract_model(model_config)
            if model_config is None:
                return

            self.model_configs[model_id].update(model_config)

        if model_config["type"] == "yolov5":
            from .yolov5 import YOLOv5

            try:
                model_config["model"] = YOLOv5(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                return
        elif model_config["type"] == "yolov8":
            from .yolov8 import YOLOv8

            try:
                model_config["model"] = YOLOv8(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                return
        elif model_config["type"] == "yolov11":
            from .yolov11 import YOLO11

            try:
                model_config["model"] = YOLO11(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                return
        elif model_config["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            try:
                model_config["model"] = SegmentAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
            except Exception as e:  # noqa
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return

            # Request next files for prediction
            self.request_next_files_requested.emit()
        else:
            raise Exception(f"Unknown model type: {model_config['type']}")

        self.loaded_model_config = model_config
        return self.loaded_model_config

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] != "segment_anything"
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_marks(marks)

    def unload_model(self):
        """Unload model"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None

    def predict_shapes(self, image, filename=None):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        if self.loaded_model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            self.prediction_finished.emit()
            return
        try:
            auto_labeling_result = self.loaded_model_config["model"].predict_shapes(
                image, filename
            )
            self.new_auto_labeling_result.emit(auto_labeling_result)
        except Exception as e:  # noqa
            print(f"Error in predict_shapes: {e}")
            self.new_model_status.emit(
                self.tr("Error in model prediction. Please check the model.")
            )
        self.new_model_status.emit(
            self.tr("Finished inferencing AI model. Check the result.")
        )
        self.prediction_finished.emit()

    @pyqtSlot()
    def predict_shapes_threading(self, image, filename=None):
        """Predict shapes.
        This function starts a thread to run the prediction.
        """
        if self.loaded_model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            return
        self.new_model_status.emit(self.tr("Inferencing AI model. Please wait..."))
        self.prediction_started.emit()

        with self.model_execution_thread_lock:
            if (
                self.model_execution_thread is not None
                and self.model_execution_thread.isRunning()
            ):
                self.new_model_status.emit(
                    self.tr(
                        "Another model is being executed. Please wait for it to finish."
                    )
                )
                self.prediction_finished.emit()
                return

            self.model_execution_thread = QThread()
            self.model_execution_worker = GenericWorker(
                self.predict_shapes, image, filename
            )
            self.model_execution_worker.finished.connect(
                self.model_execution_thread.quit
            )
            self.model_execution_worker.moveToThread(self.model_execution_thread)
            self.model_execution_thread.started.connect(self.model_execution_worker.run)
            self.model_execution_thread.start()

    def on_next_files_changed(self, next_files):
        """Run prediction on next files in advance to save inference time later"""
        if self.loaded_model_config is None:
            return

        # Currently only segment_anything model supports this feature
        if self.loaded_model_config["type"] != "segment_anything":
            return

        self.loaded_model_config["model"].on_next_files_changed(next_files)
