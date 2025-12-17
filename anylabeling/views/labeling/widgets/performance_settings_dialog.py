"""Performance settings dialog for configuring performance-related options."""

import logging

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from anylabeling.config import get_config, save_config

logger = logging.getLogger(__name__)


class PerformanceSettingsDialog(QDialog):
    """Dialog for configuring performance settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Performance Settings"))
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Set up the UI elements."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Extension Status Group
        status_group = QGroupBox(self.tr("Extension Status"))
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)

        # Check Cython extensions
        cython_available = False
        try:
            from anylabeling.extensions import extensions_available

            cython_available = extensions_available()
        except ImportError:
            pass

        cython_status = (
            self.tr("✓ Cython extensions available")
            if cython_available
            else self.tr("✗ Cython extensions not available")
        )
        self.cython_label = QLabel(cython_status)
        status_layout.addWidget(self.cython_label)

        # Check Rust extensions
        rust_available = False
        try:
            from anylabeling.rust_extensions import rust_available as check_rust

            rust_available = check_rust()
        except ImportError:
            pass

        rust_status = (
            self.tr("✓ Rust extensions available")
            if rust_available
            else self.tr("✗ Rust extensions not available")
        )
        self.rust_label = QLabel(rust_status)
        status_layout.addWidget(self.rust_label)

        # Add help text
        help_text = QLabel(
            self.tr(
                "Extensions provide significant performance improvements. "
                "See docs/building_extensions.md for installation instructions."
            )
        )
        help_text.setWordWrap(True)
        status_layout.addWidget(help_text)

        layout.addWidget(status_group)

        # Inference Settings Group
        inference_group = QGroupBox(self.tr("Inference Settings"))
        inference_layout = QVBoxLayout()
        inference_group.setLayout(inference_layout)

        # Backend selection
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel(self.tr("Backend:")))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem(self.tr("Auto (recommended)"), "auto")
        self.backend_combo.addItem(self.tr("ONNX - CPU"), "onnx-cpu")
        self.backend_combo.addItem(self.tr("ONNX - GPU"), "onnx-gpu")
        self.backend_combo.addItem(self.tr("Ultralytics"), "ultralytics")
        self.backend_combo.addItem(self.tr("TensorRT (if available)"), "tensorrt")
        backend_layout.addWidget(self.backend_combo)
        backend_layout.addStretch()
        inference_layout.addLayout(backend_layout)

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel(self.tr("Batch size:")))
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setMinimum(1)
        self.batch_size_spinner.setMaximum(16)
        self.batch_size_spinner.setValue(4)
        batch_layout.addWidget(self.batch_size_spinner)
        batch_layout.addWidget(QLabel(self.tr("(higher = faster for multiple images)")))
        batch_layout.addStretch()
        inference_layout.addLayout(batch_layout)

        layout.addWidget(inference_group)

        # Threading Settings Group
        threading_group = QGroupBox(self.tr("Threading Settings"))
        threading_layout = QVBoxLayout()
        threading_group.setLayout(threading_layout)

        # Worker thread count
        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel(self.tr("Worker threads:")))
        self.worker_threads_spinner = QSpinBox()
        self.worker_threads_spinner.setMinimum(1)
        self.worker_threads_spinner.setMaximum(16)
        self.worker_threads_spinner.setValue(8)
        worker_layout.addWidget(self.worker_threads_spinner)
        worker_layout.addWidget(QLabel(self.tr("(for parallel processing)")))
        worker_layout.addStretch()
        threading_layout.addLayout(worker_layout)

        layout.addWidget(threading_group)

        # Caching Settings Group
        caching_group = QGroupBox(self.tr("Caching Settings"))
        caching_layout = QVBoxLayout()
        caching_group.setLayout(caching_layout)

        # Image cache size
        cache_size_layout = QVBoxLayout()
        cache_size_layout.addWidget(QLabel(self.tr("Image cache size:")))

        cache_slider_layout = QHBoxLayout()
        self.cache_size_slider = QSlider(Qt.Horizontal)
        self.cache_size_slider.setMinimum(128)
        self.cache_size_slider.setMaximum(2048)
        self.cache_size_slider.setValue(512)
        self.cache_size_slider.setTickPosition(QSlider.TicksBelow)
        self.cache_size_slider.setTickInterval(256)
        self.cache_size_value_label = QLabel("512 MB")
        self.cache_size_slider.valueChanged.connect(self.update_cache_size_label)
        cache_slider_layout.addWidget(self.cache_size_slider)
        cache_slider_layout.addWidget(self.cache_size_value_label)
        cache_size_layout.addLayout(cache_slider_layout)

        caching_layout.addLayout(cache_size_layout)

        # Result caching
        self.result_caching_checkbox = QCheckBox(
            self.tr("Enable result caching (speeds up repeated operations)")
        )
        self.result_caching_checkbox.setChecked(True)
        caching_layout.addWidget(self.result_caching_checkbox)

        layout.addWidget(caching_group)

        # Pre-loading Settings Group
        preload_group = QGroupBox(self.tr("Pre-loading Settings"))
        preload_layout = QVBoxLayout()
        preload_group.setLayout(preload_layout)

        # Pre-loading enable/disable
        self.preload_enabled_checkbox = QCheckBox(
            self.tr("Enable pre-loading (loads next images in background)")
        )
        self.preload_enabled_checkbox.setChecked(True)
        self.preload_enabled_checkbox.toggled.connect(self.on_preload_toggled)
        preload_layout.addWidget(self.preload_enabled_checkbox)

        # Pre-load count
        preload_count_layout = QHBoxLayout()
        preload_count_layout.addWidget(QLabel(self.tr("Pre-load count:")))
        self.preload_count_spinner = QSpinBox()
        self.preload_count_spinner.setMinimum(1)
        self.preload_count_spinner.setMaximum(10)
        self.preload_count_spinner.setValue(3)
        preload_count_layout.addWidget(self.preload_count_spinner)
        preload_count_layout.addWidget(QLabel(self.tr("images")))
        preload_count_layout.addStretch()
        preload_layout.addLayout(preload_count_layout)

        layout.addWidget(preload_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.reset_button = QPushButton(self.tr("Reset to Defaults"))
        self.reset_button.clicked.connect(self.on_reset)
        button_layout.addWidget(self.reset_button)

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        self.apply_button = QPushButton(self.tr("Apply"))
        self.apply_button.clicked.connect(self.on_apply)
        button_layout.addWidget(self.apply_button)

        layout.addLayout(button_layout)

    def update_cache_size_label(self, value):
        """Update the cache size value label."""
        self.cache_size_value_label.setText(self.tr("%d MB") % value)

    def on_preload_toggled(self, enabled):
        """Handle pre-load checkbox toggle."""
        self.preload_count_spinner.setEnabled(enabled)

    def load_settings(self):
        """Load settings from config."""
        config = get_config()
        perf_config = config.get("performance", {})

        # Load inference settings
        backend = perf_config.get("backend", "auto")
        index = self.backend_combo.findData(backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)

        batch_size = perf_config.get("batch_size", 4)
        self.batch_size_spinner.setValue(batch_size)

        # Load threading settings
        worker_threads = perf_config.get("num_worker_threads", 8)
        self.worker_threads_spinner.setValue(worker_threads)

        # Load caching settings
        cache_size_mb = perf_config.get("image_cache_size_mb", 512)
        self.cache_size_slider.setValue(cache_size_mb)

        result_caching = perf_config.get("enable_result_caching", True)
        self.result_caching_checkbox.setChecked(result_caching)

        # Load pre-loading settings
        preload_enabled = perf_config.get("preload_enabled", True)
        self.preload_enabled_checkbox.setChecked(preload_enabled)

        preload_count = perf_config.get("preload_count", 3)
        self.preload_count_spinner.setValue(preload_count)

        # Update preload spinner enabled state
        self.preload_count_spinner.setEnabled(preload_enabled)

    def save_settings(self):
        """Save settings to config."""
        config = get_config()

        if "performance" not in config:
            config["performance"] = {}

        # Save inference settings
        config["performance"]["backend"] = self.backend_combo.currentData()
        config["performance"]["batch_size"] = self.batch_size_spinner.value()

        # Save threading settings
        config["performance"]["num_worker_threads"] = self.worker_threads_spinner.value()

        # Save caching settings
        config["performance"]["image_cache_size_mb"] = self.cache_size_slider.value()
        config["performance"]["enable_result_caching"] = (
            self.result_caching_checkbox.isChecked()
        )

        # Save pre-loading settings
        config["performance"]["preload_enabled"] = (
            self.preload_enabled_checkbox.isChecked()
        )
        config["performance"]["preload_count"] = self.preload_count_spinner.value()

        save_config(config)

    def on_reset(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            self.tr("Reset Settings"),
            self.tr("Are you sure you want to reset all performance settings to defaults?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Set default values
            self.backend_combo.setCurrentIndex(0)  # Auto
            self.batch_size_spinner.setValue(4)
            self.worker_threads_spinner.setValue(8)
            self.cache_size_slider.setValue(512)
            self.result_caching_checkbox.setChecked(True)
            self.preload_enabled_checkbox.setChecked(True)
            self.preload_count_spinner.setValue(3)

    def on_apply(self):
        """Apply settings and close dialog."""
        self.save_settings()

        QMessageBox.information(
            self,
            self.tr("Settings Saved"),
            self.tr("Performance settings have been saved. Some changes may require restarting the application."),
        )

        self.accept()
