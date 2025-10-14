"""
Pytest configuration and fixtures for Comfy Commander tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import io

from comfy_commander import Workflow, ComfyOutput


@pytest.fixture
def example_api_workflow_file_path():
    """Path to example API workflow JSON file."""
    return Path(__file__).parent / "fixtures" / "flux_dev_checkpoint_example_api.json"


@pytest.fixture
def example_standard_workflow_file_path():
    """Path to example standard workflow JSON file."""
    return Path(__file__).parent / "fixtures" / "flux_dev_checkpoint_example_standard.json"


@pytest.fixture
def example_image_file_path():
    """Path to example image file with workflow metadata."""
    return Path(__file__).parent / "fixtures" / "flux_dev_checkpoint_example_image.png"


@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    api_json = {
        "1": {
            "class_type": "KSampler",
            "_meta": {"title": "Test Sampler"},
            "inputs": {"seed": 123, "steps": 20}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Test Text Encode"},
            "inputs": {"text": "test prompt"}
        }
    }
    gui_json = {
        "nodes": [
            {"id": 1, "type": "KSampler", "widgets_values": [123, False, 20, 7.0]},
            {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["test prompt"]}
        ],
        "links": []
    }
    return Workflow(api_json=api_json, gui_json=gui_json)


@pytest.fixture
def sample_comfy_output():
    """Create a sample ComfyOutput for testing."""
    # Create a simple test image
    test_image = Image.new('RGB', (50, 50), color='blue')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()
    
    return ComfyOutput(
        data=img_data,
        filename="test_output.png",
        subfolder="output",
        type="output"
    )


@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='green')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_image.save(tmp_file.name, format='PNG')
        yield tmp_file.name
    
    # Cleanup
    if os.path.exists(tmp_file.name):
        try:
            os.unlink(tmp_file.name)
        except PermissionError:
            # On Windows, sometimes the file is still locked
            pass


@pytest.fixture
def temp_workflow_file():
    """Create a temporary workflow file for testing."""
    workflow_data = {
        "1": {
            "class_type": "KSampler",
            "inputs": {"seed": 456, "steps": 25}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        import json
        json.dump(workflow_data, tmp_file)
        yield tmp_file.name
    
    # Cleanup
    if os.path.exists(tmp_file.name):
        try:
            os.unlink(tmp_file.name)
        except PermissionError:
            # On Windows, sometimes the file is still locked
            pass
