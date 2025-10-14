"""
Tests for core Workflow functionality including loading from files and images.
"""

import pytest
import tempfile
import os
import json
from PIL import Image
import io

from comfy_commander import Workflow, ComfyOutput


class TestWorkflowCore:
    """Test core Workflow functionality."""

    def test_can_load_workflow_from_example_image(self, snapshot, example_image_file_path):
        """Test loading workflow from image with metadata."""
        workflow = Workflow.from_image(example_image_file_path)
        # Both formats should be populated when loading from image
        assert workflow.api_json is not None
        assert workflow.gui_json is not None
        workflow.api_json == snapshot
        workflow.gui_json == snapshot
    
    def test_can_load_standard_workflow_from_file(self, example_standard_workflow_file_path):
        """Test that loading a standard workflow file only populates gui_json."""
        workflow = Workflow.from_file(example_standard_workflow_file_path)
        # Standard workflow should only have GUI data
        assert workflow.gui_json is not None
        assert workflow.api_json is None
        assert "nodes" in workflow.gui_json
        assert "links" in workflow.gui_json
    
    def test_can_load_api_workflow_from_file(self, example_api_workflow_file_path):
        """Test that loading an API workflow file only populates api_json."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        # API workflow should only have API data
        assert workflow.api_json is not None
        assert workflow.gui_json is None
        assert "6" in workflow.api_json  # Should have nodes
        assert "class_type" in workflow.api_json["6"]

    def test_workflow_from_image_with_metadata(self):
        """Test loading a workflow from an image with embedded metadata."""
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create a test workflow
        test_workflow = Workflow(
            api_json={"1": {"class_type": "TestNode", "inputs": {"test": "value"}}},
            gui_json={"nodes": [{"id": 1, "type": "TestNode"}]}
        )
        
        # Create ComfyOutput with workflow reference
        comfy_output = ComfyOutput(
            data=img_data,
            filename="test_workflow_roundtrip.png",
            subfolder="output",
            type="output"
        )
        comfy_output._workflow = test_workflow
        
        # Save the image with metadata
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            comfy_output.save(tmp_path)
            
            # Load the workflow back from the image
            loaded_workflow = Workflow.from_image(tmp_path)
            
            # Verify the workflow was loaded correctly
            assert loaded_workflow.api_json == test_workflow.api_json
            assert loaded_workflow.gui_json == test_workflow.gui_json
            
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_workflow_from_image_no_metadata(self):
        """Test loading a workflow from an image without metadata raises error."""
        # Create a simple test image without metadata
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Save the image without metadata
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Write the image data directly
            with open(tmp_path, 'wb') as f:
                f.write(img_data)
            
            # Try to load workflow from image without metadata
            with pytest.raises(ValueError, match="No ComfyUI workflow metadata found"):
                Workflow.from_image(tmp_path)
            
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass
