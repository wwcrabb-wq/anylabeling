import logging
import os
import yaml
import socket
import ssl
from abc import abstractmethod

from PyQt5.QtCore import QCoreApplication, QFile, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage

from .types import AutoLabelingResult
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError
from anylabeling.utils.image_cache import ImageCache
from anylabeling.config import get_config

# Prevent issue when downloading models behind a proxy
os.environ["no_proxy"] = "*"

socket.setdefaulttimeout(240)  # Prevent timeout when downloading models


ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Prevent issue when downloading models behind a proxy


class PreloadWorker(QObject):
    """Worker for pre-loading images in background thread."""
    
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, file_paths, image_cache):
        super().__init__()
        self.file_paths = file_paths
        self.image_cache = image_cache
        self.is_cancelled = False
        
    def run(self):
        """Pre-load images into cache."""
        try:
            for file_path in self.file_paths:
                if self.is_cancelled:
                    break
                    
                # Check if already in cache
                if file_path in self.image_cache:
                    continue
                
                try:
                    # Load image
                    from PIL import Image
                    import numpy as np
                    
                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_array = np.array(img)
                    
                    # Add to cache
                    self.image_cache.put(file_path, img_array)
                    logging.debug(f"Pre-loaded image: {file_path}")
                    
                except Exception as e:
                    logging.debug(f"Failed to pre-load {file_path}: {e}")
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            
    def cancel(self):
        """Cancel pre-loading."""
        self.is_cancelled = True


class Model(QObject):
    BASE_DOWNLOAD_URL = "https://github.com/vietanhdev/anylabeling-assets/raw/main/"

    class Meta(QObject):
        required_config_names = []
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__()
        self.on_message = on_message
        # Load and check config
        if isinstance(model_config, str):
            if not os.path.isfile(model_config):
                raise FileNotFoundError(
                    QCoreApplication.translate(
                        "Model", "Config file not found: {model_config}"
                    ).format(model_config=model_config)
                )
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            self.config = model_config
        else:
            raise ValueError(
                QCoreApplication.translate(
                    "Model", "Unknown config type: {type}"
                ).format(type=type(model_config))
            )
        self.check_missing_config(
            config_names=self.Meta.required_config_names,
            config=self.config,
        )
        self.output_mode = self.Meta.default_output_mode
        # Store config_file path if provided in config
        self.config_file = self.config.get("config_file", None)
        
        # Initialize pre-loading
        self.preload_worker = None
        self.preload_thread = None
        self.image_cache = None
        self._init_image_cache()

    def get_required_widgets(self):
        """
        Get required widgets for showing in UI
        """
        return self.Meta.widgets

    def get_model_abs_path(self, model_config, model_path_field_name):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        config_folder = os.path.dirname(model_config["config_file"])
        model_path = model_config[model_path_field_name]
        if os.path.isfile(os.path.join(config_folder, model_path)):
            model_abs_path = os.path.abspath(os.path.join(config_folder, model_path))
            return model_abs_path

        # Try getting model from assets folder
        home_dir = os.path.expanduser("~")
        model_abs_path = os.path.abspath(
            os.path.join(
                home_dir,
                "anylabeling_data",
                "models",
                model_config["name"],
                model_path,
            )
        )
        return model_abs_path

    def check_missing_config(self, config_names, config):
        """
        Check if config has all required config names
        """
        for name in config_names:
            if name not in config:
                raise Exception(f"Missing config: {name}")

    @abstractmethod
    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict image and return AnyLabeling shapes
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self):
        """
        Unload memory
        """
        raise NotImplementedError

    @staticmethod
    def load_image_from_filename(filename):
        """Load image from labeling file and return image data and image path."""
        label_file = os.path.splitext(filename)[0] + ".json"
        if QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                label_file = LabelFile(label_file)
            except LabelFileError as e:
                logging.error("Error reading {}: {}".format(label_file, e))
                return None, None
            image_data = label_file.image_data
        else:
            image_data = LabelFile.load_image_file(filename)
        image = QImage.fromData(image_data)
        if image.isNull():
            logging.error("Error reading {}".format(filename))
        return image

    def _init_image_cache(self):
        """Initialize image cache from config."""
        try:
            config = get_config()
            perf_config = config.get("performance", {})
            cache_size_mb = perf_config.get("image_cache_size_mb", 512)
            self.image_cache = ImageCache(max_memory_mb=cache_size_mb)
            logging.info(f"Image cache initialized with {cache_size_mb}MB")
        except Exception as e:
            logging.warning(f"Failed to initialize image cache: {e}")
            self.image_cache = ImageCache(max_memory_mb=512)  # Default fallback

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        
        Args:
            next_files: List of file paths to pre-load
        """
        # Check if pre-loading is enabled
        config = get_config()
        perf_config = config.get("performance", {})
        preload_enabled = perf_config.get("preload_enabled", True)
        
        if not preload_enabled or not next_files or not self.image_cache:
            return
        
        # Cancel previous pre-loading if active
        if self.preload_worker and self.preload_thread:
            self.preload_worker.cancel()
            self.preload_thread.quit()
            self.preload_thread.wait()
        
        # Get preload count from config
        preload_count = perf_config.get("preload_count", 3)
        files_to_preload = next_files[:preload_count]
        
        # Start new pre-loading thread
        self.preload_worker = PreloadWorker(files_to_preload, self.image_cache)
        self.preload_thread = QThread()
        self.preload_worker.moveToThread(self.preload_thread)
        
        # Connect signals
        self.preload_thread.started.connect(self.preload_worker.run)
        self.preload_worker.finished.connect(self.preload_thread.quit)
        self.preload_worker.error.connect(lambda e: logging.warning(f"Pre-loading error: {e}"))
        
        # Start thread
        self.preload_thread.start()
        logging.info(f"Started pre-loading {len(files_to_preload)} images")

    def set_output_mode(self, mode):
        """
        Set output mode
        """
        self.output_mode = mode

    def set_config_param(self, key, value, persist=False):
        """
        Update a model's runtime config parameter.
        
        Args:
            key: The config parameter key to update
            value: The new value for the parameter
            persist: If True, write the change back to the config file
        """
        # Update in-memory config
        self.config[key] = value
        
        # Optionally persist to file
        if persist and self.config_file and os.path.isfile(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    file_config = yaml.safe_load(f)
                file_config[key] = value
                with open(self.config_file, "w") as f:
                    yaml.dump(file_config, f)
            except Exception as e:
                logging.warning(f"Failed to persist config change: {e}")
        
        # Call hook for subclasses
        self._on_config_param_changed(key, value)

    def _on_config_param_changed(self, key, value):
        """
        Hook method called when a config parameter is changed.
        Subclasses may override to react to config changes.
        """
        pass
