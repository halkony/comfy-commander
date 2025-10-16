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

    def test_load_api_json_from_file(self):
        """Test loading API JSON data from a file."""
        # Create test API JSON data
        api_data = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": 123, "steps": 20, "cfg": 7.0}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "test prompt"}
            }
        }
        
        # Create temporary file with API JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(api_data, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Create empty workflow
            workflow = Workflow(api_json=None, gui_json=None)
            assert workflow.api_json is None
            
            # Load API JSON from file
            workflow.load_api_json(tmp_path)
            
            # Verify the data was loaded correctly
            assert workflow.api_json is not None
            assert workflow.api_json == api_data
            assert "1" in workflow.api_json
            assert "2" in workflow.api_json
            assert workflow.api_json["1"]["class_type"] == "KSampler"
            assert workflow.api_json["2"]["class_type"] == "CLIPTextEncode"
            
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_load_gui_json_from_file(self):
        """Test loading GUI JSON data from a file."""
        # Create test GUI JSON data
        gui_data = {
            "nodes": [
                {
                    "id": 1,
                    "type": "KSampler",
                    "widgets_values": [123, False, 20, 7.0]
                },
                {
                    "id": 2,
                    "type": "CLIPTextEncode",
                    "widgets_values": ["test prompt"]
                }
            ],
            "links": [
                {"from": 1, "to": 2, "from_slot": 0, "to_slot": 0}
            ]
        }
        
        # Create temporary file with GUI JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(gui_data, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Create empty workflow
            workflow = Workflow(api_json=None, gui_json=None)
            assert workflow.gui_json is None
            
            # Load GUI JSON from file
            workflow.load_gui_json(tmp_path)
            
            # Verify the data was loaded correctly
            assert workflow.gui_json is not None
            assert workflow.gui_json == gui_data
            assert "nodes" in workflow.gui_json
            assert "links" in workflow.gui_json
            assert len(workflow.gui_json["nodes"]) == 2
            assert len(workflow.gui_json["links"]) == 1
            
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_load_api_json_file_not_found(self):
        """Test that load_api_json raises FileNotFoundError for non-existent file."""
        workflow = Workflow(api_json=None, gui_json=None)
        
        with pytest.raises(FileNotFoundError):
            workflow.load_api_json("non_existent_file.json")

    def test_load_gui_json_file_not_found(self):
        """Test that load_gui_json raises FileNotFoundError for non-existent file."""
        workflow = Workflow(api_json=None, gui_json=None)
        
        with pytest.raises(FileNotFoundError):
            workflow.load_gui_json("non_existent_file.json")

    def test_load_api_json_invalid_json(self):
        """Test that load_api_json raises JSONDecodeError for invalid JSON."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write("invalid json content {")
            tmp_path = tmp_file.name
        
        try:
            workflow = Workflow(api_json=None, gui_json=None)
            
            with pytest.raises(json.JSONDecodeError):
                workflow.load_api_json(tmp_path)
                
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_load_gui_json_invalid_json(self):
        """Test that load_gui_json raises JSONDecodeError for invalid JSON."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write("invalid json content {")
            tmp_path = tmp_file.name
        
        try:
            workflow = Workflow(api_json=None, gui_json=None)
            
            with pytest.raises(json.JSONDecodeError):
                workflow.load_gui_json(tmp_path)
                
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_load_both_api_and_gui_json(self):
        """Test loading both API and GUI JSON into the same workflow."""
        # Create test data
        api_data = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": 456, "steps": 25}
            }
        }
        
        gui_data = {
            "nodes": [
                {
                    "id": 1,
                    "type": "KSampler",
                    "widgets_values": [456, False, 25, 7.0]
                }
            ],
            "links": []
        }
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='_api.json', delete=False) as api_file:
            json.dump(api_data, api_file)
            api_path = api_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='_gui.json', delete=False) as gui_file:
            json.dump(gui_data, gui_file)
            gui_path = gui_file.name
        
        try:
            # Create empty workflow
            workflow = Workflow(api_json=None, gui_json=None)
            
            # Load both JSON files
            workflow.load_api_json(api_path)
            workflow.load_gui_json(gui_path)
            
            # Verify both were loaded correctly
            assert workflow.api_json == api_data
            assert workflow.gui_json == gui_data
            assert workflow.api_json["1"]["class_type"] == "KSampler"
            assert len(workflow.gui_json["nodes"]) == 1
            
        finally:
            for path in [api_path, gui_path]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except PermissionError:
                        # On Windows, sometimes the file is still locked
                        pass
