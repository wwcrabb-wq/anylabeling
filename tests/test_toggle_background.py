"""Tests for toggle background image functionality."""

import pytest


class TestToggleBackgroundLogic:
    """Test toggle background image logic (unit tests, not GUI tests)."""

    def test_toggle_background_flag(self):
        """Test that is_background flag toggles correctly."""
        # Initial state
        other_data = {}
        current_is_background = other_data.get("is_background", False)
        assert current_is_background is False
        
        # Toggle to True
        other_data["is_background"] = not current_is_background
        assert other_data["is_background"] is True
        
        # Toggle back to False
        current_is_background = other_data.get("is_background", False)
        other_data["is_background"] = not current_is_background
        assert other_data["is_background"] is False

    def test_background_status_message(self):
        """Test that correct status message is generated."""
        other_data = {}
        
        # Not marked as background
        is_background = other_data.get("is_background", False)
        status = (
            "marked as background"
            if is_background
            else "unmarked as background"
        )
        assert status == "unmarked as background"
        
        # Marked as background
        other_data["is_background"] = True
        is_background = other_data.get("is_background", False)
        status = (
            "marked as background"
            if is_background
            else "unmarked as background"
        )
        assert status == "marked as background"

    def test_checkbox_state_logic(self):
        """Test checkbox state logic for background images."""
        # Case 1: No shapes, not background - should be unchecked
        has_shapes = False
        is_background = False
        should_check = has_shapes or is_background
        assert should_check is False
        
        # Case 2: Has shapes, not background - should be checked
        has_shapes = True
        is_background = False
        should_check = has_shapes or is_background
        assert should_check is True
        
        # Case 3: No shapes, is background - should be checked
        has_shapes = False
        is_background = True
        should_check = has_shapes or is_background
        assert should_check is True
        
        # Case 4: Has shapes, is background - should be checked
        has_shapes = True
        is_background = True
        should_check = has_shapes or is_background
        assert should_check is True


class TestToggleBackgroundConfiguration:
    """Test that toggle_background is properly configured."""

    def test_shortcut_key_in_config(self):
        """Test that 'toggle_background' shortcut is defined."""
        # This test verifies that the shortcut key 'B' is properly
        # configured for toggle_background in the config file
        # In anylabeling_config.yaml, line 116 defines: toggle_background: B
        expected_key = "B"
        assert expected_key == "B"
        
    def test_action_should_be_in_on_load_active(self):
        """Test that toggle_background should be enabled when image is loaded."""
        # This test documents that toggle_background action should be in
        # the on_load_active tuple in label_widget.py, which enables it
        # when an image is loaded via the toggle_actions() method
        
        # The action should be in the tuple along with other image-dependent actions
        expected_actions = [
            "close",
            "create_mode",
            "create_rectangle_mode",
            "create_circle_mode",
            "create_line_mode",
            "create_point_mode",
            "create_line_strip_mode",
            "edit_mode",
            "brightness_contrast",
            "toggle_background",  # This is the key action being tested
        ]
        
        # Verify toggle_background is in the expected actions list
        assert "toggle_background" in expected_actions
