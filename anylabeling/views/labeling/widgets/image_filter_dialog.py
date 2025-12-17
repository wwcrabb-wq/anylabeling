"""Image filter dialog widget for filtering images based on YOLO detections."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np

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
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QScrollArea,
    QGridLayout,
    QWidget,
    QSpinBox,
    QFileDialog,
    QApplication,
)
from PyQt5.QtCore import Qt

from anylabeling.config import get_config, save_config

logger = logging.getLogger(__name__)


class FilterWorker(QObject):
    """Worker for filtering images in background thread with parallel processing."""

    progress = pyqtSignal(int, int, int)  # current, total, matched
    finished = pyqtSignal(list)  # filtered image list
    error = pyqtSignal(str)

    def __init__(
        self,
        image_paths,
        model_manager,
        min_confidence,
        max_confidence=1.0,
        selected_classes=None,
        count_mode="any",
        count_value=1,
        max_workers=None,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.model_manager = model_manager
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.selected_classes = selected_classes  # None means any class
        self.count_mode = count_mode  # "any", "at_least", "exactly", "at_most"
        self.count_value = count_value
        self.is_cancelled = False
        # Use configurable worker threads (default: min(8, cpu_count()))
        self.max_workers = (
            max_workers if max_workers is not None else min(8, cpu_count())
        )
        # Thread lock to serialize model inference (prevent concurrent access)
        self._model_lock = threading.Lock()

    def _process_single_image(self, image_path):
        """Process a single image and return if it has detections."""
        try:
            from PIL import Image
            import numpy as np

            # Load image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array for model
            image_array = np.array(image)

            # Run prediction with lock to prevent concurrent model access
            # This prevents thread-safety issues with Ultralytics models
            with self._model_lock:
                result = self.model_manager.loaded_model_config["model"].predict_shapes(
                    image_array, image_path
                )

            # Collect matching detections
            matching_detections = []
            if result and hasattr(result, "shapes"):
                for shape in result.shapes:
                    confidence = None
                    label = None
                    
                    # Extract confidence score
                    if hasattr(shape, "score") and shape.score is not None:
                        confidence = shape.score
                    # Also try to extract from description
                    elif hasattr(shape, "description") and shape.description:
                        try:
                            # Expected format: "label XX%" where XX is confidence percentage
                            if (
                                isinstance(shape.description, str)
                                and "%" in shape.description
                            ):
                                conf_str = shape.description.split("%")[0].split()[-1]
                                confidence = float(conf_str) / 100.0
                        except (ValueError, IndexError, AttributeError) as e:
                            logger.debug(
                                "Could not parse confidence from shape description '%s': %s",
                                shape.description,
                                e,
                            )
                    
                    # Extract label
                    if hasattr(shape, "label") and shape.label:
                        label = shape.label
                    elif hasattr(shape, "description") and shape.description:
                        # Try to extract label from description (format: "label XX%")
                        try:
                            if isinstance(shape.description, str):
                                label = shape.description.split()[0] if " " in shape.description else shape.description
                        except (IndexError, AttributeError):
                            pass
                    
                    # Check if detection meets criteria
                    # 1. Check confidence threshold
                    if confidence is not None:
                        if not (self.min_confidence <= confidence <= self.max_confidence):
                            continue
                    else:
                        # If no confidence info, include by default
                        logger.debug(
                            "No confidence score found for shape, including by default"
                        )
                    
                    # 2. Check class filter
                    if self.selected_classes:  # If specific classes are selected
                        if label and label in self.selected_classes:
                            matching_detections.append((label, confidence))
                    else:
                        # No class filter, include all
                        matching_detections.append((label, confidence))
            
            # Check detection count
            detection_count = len(matching_detections)
            meets_count_criteria = False
            
            if self.count_mode == "any":
                meets_count_criteria = detection_count > 0
            elif self.count_mode == "at_least":
                meets_count_criteria = detection_count >= self.count_value
            elif self.count_mode == "exactly":
                meets_count_criteria = detection_count == self.count_value
            elif self.count_mode == "at_most":
                meets_count_criteria = 0 < detection_count <= self.count_value
            else:
                # Default to any
                meets_count_criteria = detection_count > 0
            
            return (image_path, meets_count_criteria, detection_count, matching_detections)

        except Exception as e:
            logger.warning("Error processing %s: %s", image_path, e)
            return (image_path, False, 0, [])

    def run(self):
        """Run filtering process with parallel processing."""
        try:
            filtered_images = []
            completed = 0
            total = len(self.image_paths)

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self._process_single_image, path): path
                    for path in self.image_paths
                }

                # Process completed tasks
                for future in as_completed(future_to_path):
                    if self.is_cancelled:
                        # Cancel remaining tasks
                        for f in future_to_path:
                            f.cancel()
                        break

                    try:
                        image_path, has_detection, detection_count, detections = future.result()
                        if has_detection:
                            filtered_images.append(image_path)
                    except Exception as e:
                        logger.error("Error processing image: %s", e)

                    completed += 1
                    # Emit progress (thread-safe)
                    self.progress.emit(completed, total, len(filtered_images))

            if not self.is_cancelled:
                self.finished.emit(filtered_images)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        """Cancel the filtering process."""
        self.is_cancelled = True


class ImageFilterDialog(QDialog):
    """Dialog for filtering images based on detections."""

    def __init__(self, parent=None, model_manager=None, image_paths=None, folder_path=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.image_paths = image_paths or []
        self.folder_path = folder_path  # Store folder path for caching
        self.filtered_images = None
        self.filtered_results = {}  # Store detailed results for export
        self.worker = None
        self.worker_thread = None
        self.thumbnail_widgets = []  # Store thumbnail widgets
        self.cache_used = False  # Track if results came from cache

        # Initialize filter result cache
        from anylabeling.utils.image_cache import FilterResultCache

        self.result_cache = FilterResultCache()

        self.setWindowTitle(self.tr("Image Filter Options"))
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
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
        self.max_confidence_slider.valueChanged.connect(
            self.update_max_confidence_label
        )
        max_slider_layout.addWidget(self.max_confidence_slider)
        max_slider_layout.addWidget(self.max_confidence_value_label)
        filter_options_layout.addLayout(max_slider_layout)

        # Class filtering section
        class_filter_label = QLabel(self.tr("Classes to detect:"))
        filter_options_layout.addWidget(class_filter_label)

        self.any_class_radio = QRadioButton(self.tr("Any class (all detections)"))
        self.any_class_radio.setChecked(True)
        self.selected_classes_radio = QRadioButton(self.tr("Selected classes only:"))

        self.class_mode_group = QButtonGroup()
        self.class_mode_group.addButton(self.any_class_radio)
        self.class_mode_group.addButton(self.selected_classes_radio)

        filter_options_layout.addWidget(self.any_class_radio)
        filter_options_layout.addWidget(self.selected_classes_radio)

        # Class selection list
        self.class_list_widget = QListWidget()
        self.class_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.class_list_widget.setMaximumHeight(120)
        self.class_list_widget.setEnabled(False)
        filter_options_layout.addWidget(self.class_list_widget)

        # Select All / Select None buttons
        class_button_layout = QHBoxLayout()
        self.select_all_button = QPushButton(self.tr("Select All"))
        self.select_all_button.clicked.connect(self.on_select_all_classes)
        self.select_all_button.setEnabled(False)
        self.select_none_button = QPushButton(self.tr("Select None"))
        self.select_none_button.clicked.connect(self.on_select_none_classes)
        self.select_none_button.setEnabled(False)
        class_button_layout.addWidget(self.select_all_button)
        class_button_layout.addWidget(self.select_none_button)
        class_button_layout.addStretch()
        filter_options_layout.addLayout(class_button_layout)

        # Connect class mode radio to enable/disable class list
        self.any_class_radio.toggled.connect(self.on_class_mode_changed)

        # Populate class list from model
        self.populate_class_list()

        # Detection count filter section
        count_filter_label = QLabel(self.tr("Detection count filter:"))
        filter_options_layout.addWidget(count_filter_label)

        count_layout = QHBoxLayout()
        self.count_mode_combo = QComboBox()
        self.count_mode_combo.addItem(self.tr("Any count"), "any")
        self.count_mode_combo.addItem(self.tr("At least"), "at_least")
        self.count_mode_combo.addItem(self.tr("Exactly"), "exactly")
        self.count_mode_combo.addItem(self.tr("At most"), "at_most")
        count_layout.addWidget(self.count_mode_combo)

        self.count_value_spinner = QSpinBox()
        self.count_value_spinner.setMinimum(1)
        self.count_value_spinner.setMaximum(100)
        self.count_value_spinner.setValue(1)
        self.count_value_spinner.setEnabled(False)
        count_layout.addWidget(self.count_value_spinner)
        count_layout.addWidget(QLabel(self.tr("detection(s)")))
        count_layout.addStretch()
        filter_options_layout.addLayout(count_layout)

        # Connect count mode combo to enable/disable spinner
        self.count_mode_combo.currentIndexChanged.connect(self.on_count_mode_changed)

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

        # Thumbnail preview section
        preview_group = QGroupBox(self.tr("Preview (max 50 thumbnails)"))
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)

        # Scroll area for thumbnails
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setMinimumHeight(150)
        self.thumbnail_scroll.setMaximumHeight(200)

        # Container widget and grid layout for thumbnails
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QGridLayout()
        self.thumbnail_layout.setSpacing(5)
        self.thumbnail_container.setLayout(self.thumbnail_layout)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)

        preview_layout.addWidget(self.thumbnail_scroll)

        self.thumbnail_status_label = QLabel(self.tr("No thumbnails to display"))
        preview_layout.addWidget(self.thumbnail_status_label)

        layout.addWidget(preview_group)

        # Cache status label
        self.cache_status_label = QLabel("")
        layout.addWidget(self.cache_status_label)

        # Buttons
        button_layout = QHBoxLayout()
        
        self.clear_cache_button = QPushButton(self.tr("Clear Cache"))
        self.clear_cache_button.clicked.connect(self.on_clear_cache)
        button_layout.addWidget(self.clear_cache_button)
        
        button_layout.addStretch()

        self.export_button = QPushButton(self.tr("Export Results"))
        self.export_button.clicked.connect(self.on_export_results)
        self.export_button.setVisible(False)  # Hidden until filtering completes
        button_layout.addWidget(self.export_button)

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_button)

        self.apply_button = QPushButton(self.tr("Apply Filter"))
        self.apply_button.clicked.connect(self.on_apply)
        button_layout.addWidget(self.apply_button)

        layout.addLayout(button_layout)

        # Load settings from config
        self.load_settings()

    def populate_class_list(self):
        """Populate the class list widget with available classes from the model."""
        self.class_list_widget.clear()

        if not self.model_manager or not self.model_manager.loaded_model_config:
            return

        # Try to get classes from the loaded model
        model = self.model_manager.loaded_model_config.get("model")
        if not model:
            return

        classes = None
        # Try different ways to get class names
        if hasattr(model, "classes"):
            classes = model.classes
        elif hasattr(model, "names"):
            classes = model.names
        elif hasattr(model, "config") and "classes" in model.config:
            classes = model.config["classes"]

        if classes:
            for class_name in classes:
                item = QListWidgetItem(class_name)
                self.class_list_widget.addItem(item)

    def on_class_mode_changed(self):
        """Handle class mode radio button change."""
        any_class_selected = self.any_class_radio.isChecked()
        self.class_list_widget.setEnabled(not any_class_selected)
        self.select_all_button.setEnabled(not any_class_selected)
        self.select_none_button.setEnabled(not any_class_selected)

    def on_select_all_classes(self):
        """Select all classes in the class list."""
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            item.setSelected(True)

    def on_select_none_classes(self):
        """Deselect all classes in the class list."""
        self.class_list_widget.clearSelection()

    def on_count_mode_changed(self):
        """Handle count mode combo box change."""
        count_mode = self.count_mode_combo.currentData()
        # Enable spinner for all modes except "any"
        self.count_value_spinner.setEnabled(count_mode != "any")

    def clear_thumbnails(self):
        """Clear all thumbnail widgets."""
        # Remove all widgets from layout
        while self.thumbnail_layout.count():
            item = self.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.thumbnail_widgets.clear()
        self.thumbnail_status_label.setText(self.tr("No thumbnails to display"))

    def add_thumbnail(self, image_path, row, col):
        """Add a thumbnail to the preview grid."""
        try:
            from PIL import Image
            from PyQt5.QtGui import QPixmap

            # Load and resize image
            img = Image.open(image_path)
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)

            # Convert to QPixmap
            img_array = np.array(img)
            from PyQt5.QtGui import QImage

            if len(img_array.shape) == 2:  # Grayscale
                height, width = img_array.shape
                bytes_per_line = width
                q_img = QImage(
                    img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8
                )
            else:  # RGB
                height, width, channel = img_array.shape
                bytes_per_line = 3 * width
                q_img = QImage(
                    img_array.data, width, height, bytes_per_line, QImage.Format_RGB888
                )

            pixmap = QPixmap.fromImage(q_img)

            # Create label for thumbnail
            thumbnail_label = QLabel()
            thumbnail_label.setPixmap(pixmap)
            thumbnail_label.setToolTip(image_path)
            thumbnail_label.setScaledContents(False)

            self.thumbnail_layout.addWidget(thumbnail_label, row, col)
            self.thumbnail_widgets.append(thumbnail_label)

        except Exception as e:
            logger.warning("Failed to create thumbnail for %s: %s", image_path, e)

    def update_thumbnails(self, filtered_images):
        """Update thumbnail preview with filtered images."""
        self.clear_thumbnails()

        if not filtered_images:
            self.thumbnail_status_label.setText(self.tr("No thumbnails to display"))
            return

        # Limit to 50 thumbnails
        max_thumbnails = 50
        display_count = min(len(filtered_images), max_thumbnails)

        # Display thumbnails in grid (10 columns)
        cols = 10
        for i in range(display_count):
            row = i // cols
            col = i % cols
            self.add_thumbnail(filtered_images[i], row, col)

        # Update status label
        if len(filtered_images) > max_thumbnails:
            self.thumbnail_status_label.setText(
                self.tr("Showing %d thumbnails (and %d more)")
                % (max_thumbnails, len(filtered_images) - max_thumbnails)
            )
        else:
            self.thumbnail_status_label.setText(
                self.tr("Showing %d thumbnail(s)") % display_count
            )

    def on_export_results(self):
        """Export filtered results to file."""
        if not self.filtered_images:
            QMessageBox.warning(
                self,
                self.tr("No Results"),
                self.tr("No filtered results to export."),
            )
            return

        # Get export directory from config or use last known
        config = get_config()
        filter_config = config.get("image_filter", {})
        last_export_dir = filter_config.get("last_export_dir", "")

        # Ask user for export format
        from PyQt5.QtWidgets import QInputDialog

        formats = ["JSON (detailed)", "TXT (paths only)", "CSV (summary)", "Copy to Clipboard"]
        format_choice, ok = QInputDialog.getItem(
            self,
            self.tr("Export Format"),
            self.tr("Choose export format:"),
            formats,
            0,
            False,
        )

        if not ok:
            return

        if format_choice == "Copy to Clipboard":
            # Copy paths to clipboard
            clipboard_text = "\n".join(self.filtered_images)
            QApplication.clipboard().setText(clipboard_text)
            QMessageBox.information(
                self,
                self.tr("Export Complete"),
                self.tr("Copied %d file paths to clipboard.") % len(self.filtered_images),
            )
            return

        # Determine file extension
        if format_choice.startswith("JSON"):
            ext = "json"
            filter_str = "JSON files (*.json)"
        elif format_choice.startswith("TXT"):
            ext = "txt"
            filter_str = "Text files (*.txt)"
        else:  # CSV
            ext = "csv"
            filter_str = "CSV files (*.csv)"

        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export Results"),
            last_export_dir or f"filter_results.{ext}",
            filter_str,
        )

        if not file_path:
            return

        # Save export directory to config
        import os

        export_dir = os.path.dirname(file_path)
        if "image_filter" not in config:
            config["image_filter"] = {}
        config["image_filter"]["last_export_dir"] = export_dir
        save_config(config)

        # Export based on format
        try:
            if format_choice.startswith("JSON"):
                self.export_json(file_path)
            elif format_choice.startswith("TXT"):
                self.export_txt(file_path)
            else:  # CSV
                self.export_csv(file_path)

            QMessageBox.information(
                self,
                self.tr("Export Complete"),
                self.tr("Results exported to:\n%s") % file_path,
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Export Error"),
                self.tr("Failed to export results: %s") % str(e),
            )

    def export_json(self, file_path):
        """Export results as JSON with detailed information."""
        import json

        export_data = {
            "total_images": len(self.image_paths),
            "filtered_images": len(self.filtered_images),
            "filter_settings": {
                "min_confidence": self.confidence_slider.value() / 100.0,
                "max_confidence": self.max_confidence_slider.value() / 100.0,
                "selected_classes": self.get_selected_classes(),
                "count_mode": self.count_mode_combo.currentData(),
                "count_value": self.count_value_spinner.value(),
            },
            "results": self.filtered_images,
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def export_txt(self, file_path):
        """Export results as plain text file with paths."""
        with open(file_path, "w") as f:
            for img_path in self.filtered_images:
                f.write(img_path + "\n")

    def export_csv(self, file_path):
        """Export results as CSV with summary information."""
        import csv

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path"])
            for img_path in self.filtered_images:
                writer.writerow([img_path])

    def get_selected_classes(self):
        """Get list of selected class names."""
        if self.any_class_radio.isChecked():
            return None  # All classes

        selected_classes = []
        for item in self.class_list_widget.selectedItems():
            selected_classes.append(item.text())
        return selected_classes if selected_classes else None

    def on_clear_cache(self):
        """Clear the filter result cache."""
        reply = QMessageBox.question(
            self,
            self.tr("Clear Cache"),
            self.tr("Are you sure you want to clear the filter result cache?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.result_cache.clear()
            QMessageBox.information(
                self,
                self.tr("Cache Cleared"),
                self.tr("Filter result cache has been cleared."),
            )
            self.cache_status_label.setText("")

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

        # Load class selection
        selected_classes = filter_config.get("selected_classes", None)
        if selected_classes:
            self.selected_classes_radio.setChecked(True)
            self.class_list_widget.setEnabled(True)
            self.select_all_button.setEnabled(True)
            self.select_none_button.setEnabled(True)
            # Select the classes
            for i in range(self.class_list_widget.count()):
                item = self.class_list_widget.item(i)
                if item.text() in selected_classes:
                    item.setSelected(True)
        else:
            self.any_class_radio.setChecked(True)

        # Load count filter settings
        count_mode = filter_config.get("count_mode", "any")
        index = self.count_mode_combo.findData(count_mode)
        if index >= 0:
            self.count_mode_combo.setCurrentIndex(index)

        count_value = filter_config.get("count_value", 1)
        self.count_value_spinner.setValue(count_value)

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

        # Save class selection
        config["image_filter"]["selected_classes"] = self.get_selected_classes()

        # Save count filter settings
        config["image_filter"]["count_mode"] = self.count_mode_combo.currentData()
        config["image_filter"]["count_value"] = self.count_value_spinner.value()

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

        # Get selected classes
        selected_classes = self.get_selected_classes()

        # Get count filter settings
        count_mode = self.count_mode_combo.currentData()
        count_value = self.count_value_spinner.value()

        # Check if result caching is enabled
        config = get_config()
        perf_config = config.get("performance", {})
        enable_caching = perf_config.get("enable_result_caching", True)

        # Try to get from cache if enabled and folder path is available
        if enable_caching and self.folder_path:
            model_name = "unknown"
            if self.model_manager and self.model_manager.loaded_model_config:
                model_name = self.model_manager.loaded_model_config.get("name", "unknown")

            cached_result = self.result_cache.get(
                self.folder_path,
                model_name,
                min_confidence,
                max_confidence,
                selected_classes,
                count_mode,
                count_value,
            )

            if cached_result:
                # Use cached results
                filtered_images = cached_result.get("filtered_images", [])
                self.filtered_images = filtered_images
                self.cache_used = True

                # Update UI
                self.update_thumbnails(filtered_images)
                self.export_button.setVisible(True)
                self.progress_bar.setValue(100)
                self.progress_label.setText(self.tr("Loaded from cache"))
                self.matched_label.setText(self.tr("Matched: %d images") % len(filtered_images))
                
                # Show cache status
                cache_stats = self.result_cache.get_stats()
                self.cache_status_label.setText(
                    self.tr("✓ Results from cache (hit rate: %.1f%%)") % (cache_stats["hit_rate"] * 100)
                )

                # Show summary
                QMessageBox.information(
                    self,
                    self.tr("Filtering Complete (Cached)"),
                    self.tr("Found %d images with detections out of %d total images.\n(Results loaded from cache)")
                    % (len(filtered_images), len(self.image_paths)),
                )

                return  # Don't start filtering

        # Clear previous thumbnails
        self.clear_thumbnails()
        self.cache_used = False
        self.cache_status_label.setText("")

        # Disable buttons during processing
        self.apply_button.setEnabled(False)
        self.no_filter_radio.setEnabled(False)
        self.filter_radio.setEnabled(False)
        self.filter_options_group.setEnabled(False)

        # Create worker and thread
        self.worker = FilterWorker(
            self.image_paths,
            self.model_manager,
            min_confidence,
            max_confidence,
            selected_classes,
            count_mode,
            count_value,
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

        # Save to cache if enabled and folder path is available
        config = get_config()
        perf_config = config.get("performance", {})
        enable_caching = perf_config.get("enable_result_caching", True)

        if enable_caching and self.folder_path:
            model_name = "unknown"
            if self.model_manager and self.model_manager.loaded_model_config:
                model_name = self.model_manager.loaded_model_config.get("name", "unknown")

            # Get filter settings
            min_confidence = self.confidence_slider.value() / 100.0
            max_confidence = self.max_confidence_slider.value() / 100.0
            selected_classes = self.get_selected_classes()
            count_mode = self.count_mode_combo.currentData()
            count_value = self.count_value_spinner.value()

            # Save to cache
            self.result_cache.put(
                self.folder_path,
                model_name,
                min_confidence,
                max_confidence,
                selected_classes,
                count_mode,
                count_value,
                filtered_images,
                len(self.image_paths),
            )

            # Update cache status
            cache_stats = self.result_cache.get_stats()
            self.cache_status_label.setText(
                self.tr("✓ Results cached (%d entries, %.1fMB)") 
                % (cache_stats["entries"], cache_stats["size_mb"])
            )

        # Update thumbnails
        self.update_thumbnails(filtered_images)

        # Show export button
        self.export_button.setVisible(True)

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
