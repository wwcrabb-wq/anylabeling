"""Image filter dialog widget for filtering images based on YOLO detections."""

import logging

from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QProgressBar,
    QSlider,
    QGroupBox,
    QComboBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from anylabeling.config import get_config, save_config

logger = logging.getLogger(__name__)


class FilterWorker(QObject):
    """Worker for filtering images in background thread."""

    progress = pyqtSignal(int, int, int)  # current, total, matched
    finished = pyqtSignal(list)  # filtered image list
    error = pyqtSignal(str)

    def __init__(self, image_paths, model_manager, min_confidence, max_confidence=1.0):
        super().__init__()
        self.image_paths = image_paths
        self.model_manager = model_manager
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.is_cancelled = False

    def run(self):
        """Run filtering process."""
        try:
            from PIL import Image
            import numpy as np

            filtered_images = []

            for idx, image_path in enumerate(self.image_paths):
                if self.is_cancelled:
                    break

                try:
                    # Load image
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Convert to numpy array for model
                    image_array = np.array(image)

                    # Run prediction
                    result = self.model_manager.loaded_model_config[
                        "model"
                    ].predict_shapes(image_array, image_path)

                    # Check if any detection meets threshold
                    has_detection = False
                    if result and hasattr(result, "shapes"):
                        for shape in result.shapes:
                            # Check if shape has score attribute and meets threshold
                            if hasattr(shape, "score") and shape.score is not None:
                                if self.min_confidence <= shape.score <= self.max_confidence:
                                    has_detection = True
                                    break
                            # Also check shape description for confidence
                            elif hasattr(shape, "description") and shape.description:
                                try:
                                    # Try to extract confidence from description
                                    # Expected format: "label XX%" where XX is confidence percentage
                                    # e.g., "person 85%", "car 92%"
                                    if (
                                        isinstance(shape.description, str)
                                        and "%" in shape.description
                                    ):
                                        conf_str = shape.description.split("%")[
                                            0
                                        ].split()[-1]
                                        confidence = float(conf_str) / 100.0
                                        if self.min_confidence <= confidence <= self.max_confidence:
                                            has_detection = True
                                            break
                                except (ValueError, IndexError, AttributeError) as e:
                                    # If can't parse confidence, log and skip this detection
                                    logger.debug(
                                        "Could not parse confidence from shape description '%s': %s",
                                        shape.description,
                                        e,
                                    )
                                    # Don't assume it passes - continue checking other shapes
                                    continue
                            else:
                                # If no score info, assume it passes threshold
                                # This is a fallback for models that don't provide confidence scores
                                # Note: This may lead to false positives with some models
                                logger.debug(
                                    "No confidence score found for shape, including by default"
                                )
                                has_detection = True
                                break

                    if has_detection:
                        filtered_images.append(image_path)

                    # Emit progress
                    self.progress.emit(
                        idx + 1, len(self.image_paths), len(filtered_images)
                    )

                except Exception as e:
                    logger.warning("Error processing %s: %s", image_path, e)
                    continue

            if not self.is_cancelled:
                self.finished.emit(filtered_images)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        """Cancel the filtering process."""
        self.is_cancelled = True


class ImageFilterDialog(QDialog):
    """Dialog for filtering images based on detections."""

    def __init__(self, parent=None, model_manager=None, image_paths=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.image_paths = image_paths or []
        self.filtered_images = None
        self.worker = None
        self.worker_thread = None

        self.setWindowTitle(self.tr("Image Filter Options"))
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI elements."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Filter mode selection
        mode_group = QGroupBox(self.tr("Filter Mode"))
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.no_filter_radio = QRadioButton(self.tr("Load all images (no filtering)"))
        self.no_filter_radio.setChecked(True)
        self.filter_radio = QRadioButton(self.tr("Filter images by detections"))

        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.no_filter_radio)
        self.mode_button_group.addButton(self.filter_radio)

        mode_layout.addWidget(self.no_filter_radio)
        mode_layout.addWidget(self.filter_radio)

        layout.addWidget(mode_group)

        # Filter options (only enabled when filter mode is selected)
        self.filter_options_group = QGroupBox(self.tr("Filter Options"))
        filter_options_layout = QVBoxLayout()
        self.filter_options_group.setLayout(filter_options_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel(self.tr("Model:")))
        self.model_combo = QComboBox()
        self.populate_model_combo()
        model_layout.addWidget(self.model_combo)
        filter_options_layout.addLayout(model_layout)

        # Minimum confidence threshold slider
        min_confidence_label = QLabel(self.tr("Minimum Confidence:"))
        filter_options_layout.addWidget(min_confidence_label)

        min_slider_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_value_label = QLabel("0.50")
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        min_slider_layout.addWidget(self.confidence_slider)
        min_slider_layout.addWidget(self.confidence_value_label)
        filter_options_layout.addLayout(min_slider_layout)

        # Maximum confidence threshold slider
        max_confidence_label = QLabel(self.tr("Maximum Confidence:"))
        filter_options_layout.addWidget(max_confidence_label)

        max_slider_layout = QHBoxLayout()
        self.max_confidence_slider = QSlider(Qt.Horizontal)
        self.max_confidence_slider.setMinimum(0)
        self.max_confidence_slider.setMaximum(100)
        self.max_confidence_slider.setValue(100)
        self.max_confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.max_confidence_slider.setTickInterval(10)
        self.max_confidence_value_label = QLabel("1.00")
        self.max_confidence_slider.valueChanged.connect(self.update_max_confidence_label)
        max_slider_layout.addWidget(self.max_confidence_slider)
        max_slider_layout.addWidget(self.max_confidence_value_label)
        filter_options_layout.addLayout(max_slider_layout)

        # Initially disable filter options
        self.filter_options_group.setEnabled(False)
        layout.addWidget(self.filter_options_group)

        # Connect radio button to enable/disable filter options
        self.no_filter_radio.toggled.connect(self.on_filter_mode_changed)

        # Progress section
        progress_group = QGroupBox(self.tr("Progress"))
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel(self.tr("Ready to filter"))
        progress_layout.addWidget(self.progress_label)

        self.matched_label = QLabel(self.tr("Matched: 0 images"))
        progress_layout.addWidget(self.matched_label)

        layout.addWidget(progress_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_button)

        self.apply_button = QPushButton(self.tr("Apply Filter"))
        self.apply_button.clicked.connect(self.on_apply)
        button_layout.addWidget(self.apply_button)

        layout.addLayout(button_layout)

        # Load settings from config
        self.load_settings()

    def load_settings(self):
        """Load filter settings from config."""
        config = get_config()
        filter_config = config.get("image_filter", {})

        # Load filter enabled state
        filter_enabled = filter_config.get("enabled", False)
        if filter_enabled:
            self.filter_radio.setChecked(True)
            self.filter_options_group.setEnabled(True)
        else:
            self.no_filter_radio.setChecked(True)
            self.filter_options_group.setEnabled(False)

        # Load confidence thresholds
        min_confidence = filter_config.get("min_confidence", 0.5)
        slider_value = int(min_confidence * 100)
        self.confidence_slider.setValue(slider_value)

        max_confidence = filter_config.get("max_confidence", 1.0)
        max_slider_value = int(max_confidence * 100)
        self.max_confidence_slider.setValue(max_slider_value)

    def save_settings(self):
        """Save filter settings to config."""
        config = get_config()

        if "image_filter" not in config:
            config["image_filter"] = {}

        # Save filter enabled state
        config["image_filter"]["enabled"] = self.filter_radio.isChecked()

        # Save confidence thresholds
        config["image_filter"]["min_confidence"] = (
            self.confidence_slider.value() / 100.0
        )
        config["image_filter"]["max_confidence"] = (
            self.max_confidence_slider.value() / 100.0
        )

        save_config(config)

    def populate_model_combo(self):
        """Populate the model combo box with available models."""
        self.model_combo.clear()

        if not self.model_manager:
            self.model_combo.addItem(self.tr("No model manager available"))
            self.model_combo.setEnabled(False)
            return

        # Check if a model is currently loaded
        if self.model_manager.loaded_model_config:
            model_name = self.model_manager.loaded_model_config.get(
                "display_name", "Current Model"
            )
            self.model_combo.addItem(
                f"{model_name} (loaded)", self.model_manager.loaded_model_config
            )
        else:
            self.model_combo.addItem(self.tr("No model loaded"))
            self.model_combo.setEnabled(False)

    def update_confidence_label(self, value):
        """Update the confidence value label."""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")

    def update_max_confidence_label(self, value):
        """Update the maximum confidence value label."""
        confidence = value / 100.0
        self.max_confidence_value_label.setText(f"{confidence:.2f}")

    def on_filter_mode_changed(self):
        """Handle filter mode change."""
        no_filter_selected = self.no_filter_radio.isChecked()
        self.filter_options_group.setEnabled(not no_filter_selected)

    def on_cancel(self):
        """Handle cancel button click."""
        if self.worker:
            self.worker.cancel()
        self.reject()

    def on_apply(self):
        """Handle apply button click."""
        # Save settings first
        self.save_settings()

        # If no filtering, just accept
        if self.no_filter_radio.isChecked():
            self.filtered_images = None  # None means no filtering
            self.accept()
            return

        # Check if there are images to filter
        if not self.image_paths:
            QMessageBox.warning(
                self,
                self.tr("No Images"),
                self.tr("No images found in the selected folder."),
            )
            return

        # Check if model is loaded
        if not self.model_manager or not self.model_manager.loaded_model_config:
            QMessageBox.warning(
                self,
                self.tr("No Model Loaded"),
                self.tr("Please load a model before filtering images."),
            )
            return

        # Start filtering process
        self.start_filtering()

    def start_filtering(self):
        """Start the filtering process in a background thread."""
        # Get confidence thresholds
        min_confidence = self.confidence_slider.value() / 100.0
        max_confidence = self.max_confidence_slider.value() / 100.0

        # Validate that min <= max
        if min_confidence > max_confidence:
            QMessageBox.warning(
                self,
                self.tr("Invalid Range"),
                self.tr(
                    "Minimum confidence (%.2f) cannot be greater than maximum confidence (%.2f)."
                )
                % (min_confidence, max_confidence),
            )
            return

        # Disable buttons during processing
        self.apply_button.setEnabled(False)
        self.no_filter_radio.setEnabled(False)
        self.filter_radio.setEnabled(False)
        self.filter_options_group.setEnabled(False)

        # Create worker and thread
        self.worker = FilterWorker(
            self.image_paths, self.model_manager, min_confidence, max_confidence
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_filtering_finished)
        self.worker.error.connect(self.on_error)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Start thread
        self.worker_thread.start()

    def on_progress(self, current, total, matched):
        """Handle progress update."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_label.setText(
                self.tr("Processing: %d/%d images") % (current, total)
            )
            self.matched_label.setText(self.tr("Matched: %d images") % matched)

    def on_filtering_finished(self, filtered_images):
        """Handle filtering completion."""
        self.filtered_images = filtered_images
        self.worker_thread.quit()
        self.worker_thread.wait()

        # Show summary
        QMessageBox.information(
            self,
            self.tr("Filtering Complete"),
            self.tr("Found %d images with detections out of %d total images.")
            % (len(filtered_images), len(self.image_paths)),
        )

        self.accept()

    def on_error(self, error_msg):
        """Handle error during filtering."""
        self.worker_thread.quit()
        self.worker_thread.wait()

        QMessageBox.critical(
            self,
            self.tr("Filtering Error"),
            self.tr("An error occurred during filtering: %s") % error_msg,
        )

        # Re-enable buttons
        self.apply_button.setEnabled(True)
        self.no_filter_radio.setEnabled(True)
        self.filter_radio.setEnabled(True)
        # Re-enable filter options if filter mode is selected
        if self.filter_radio.isChecked():
            self.filter_options_group.setEnabled(True)

    def get_filtered_images(self):
        """Return the filtered image list, or None if no filtering was applied."""
        return self.filtered_images
