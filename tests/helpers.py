"""
Test helpers for Comfy Commander tests.
"""

import tempfile
import os
from typing import Dict, Any, Optional, List
from PIL import Image
import io

from comfy_commander import Workflow, ComfyOutput


def find_gui_node_by_id(workflow: Workflow, node_id: int) -> Optional[Dict[str, Any]]:
    """Helper function to find a GUI node by its ID."""
    if workflow.gui_json is None:
        return None
    for node in workflow.gui_json["nodes"]:
        if node["id"] == node_id:
            return node
    return None


def assert_api_param_updated(workflow: Workflow, node_id: str, param_name: str, expected_value: Any) -> None:
    """Helper function to assert that an API parameter was updated."""
    assert workflow.api_json is not None, "API JSON is None, cannot check parameter"
    assert workflow.api_json[node_id]["inputs"][param_name] == expected_value


def assert_gui_widget_updated(workflow: Workflow, node_id: int, widget_index: int, expected_value: Any) -> None:
    """Helper function to assert that a GUI widget value was updated."""
    gui_node = find_gui_node_by_id(workflow, node_id)
    assert gui_node is not None, f"GUI node with ID {node_id} not found"
    assert gui_node["widgets_values"][widget_index] == expected_value


def assert_connections_preserved(workflow: Workflow, node_id: str, expected_connections: list) -> None:
    """Helper function to assert that node connections are preserved."""
    assert workflow.api_json is not None, "API JSON is None, cannot check connections"
    for connection in expected_connections:
        assert connection in workflow.api_json[node_id]["inputs"], f"Connection '{connection}' not found"


def assert_gui_connections_preserved(workflow: Workflow, node_id: int, expected_input_count: int, expected_output_count: int) -> None:
    """Helper function to assert that GUI node connections are preserved."""
    gui_node = find_gui_node_by_id(workflow, node_id)
    assert gui_node is not None, f"GUI node with ID {node_id} not found"
    assert len(gui_node["inputs"]) == expected_input_count, f"Expected {expected_input_count} inputs, got {len(gui_node['inputs'])}"
    assert len(gui_node["outputs"]) == expected_output_count, f"Expected {expected_output_count} outputs, got {len(gui_node['outputs'])}"


def create_test_image(size: tuple = (100, 100), color: str = 'red') -> bytes:
    """Create a test image and return its bytes."""
    test_image = Image.new('RGB', size, color=color)
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


def create_test_workflow() -> Workflow:
    """Create a simple test workflow."""
    api_json = {
        "1": {
            "class_type": "KSampler",
            "_meta": {"title": "Test Sampler"},
            "inputs": {"seed": 123, "steps": 20}
        }
    }
    gui_json = {
        "nodes": [
            {"id": 1, "type": "KSampler", "widgets_values": [123, False, 20, 7.0]}
        ],
        "links": []
    }
    return Workflow(api_json=api_json, gui_json=gui_json)


def create_test_comfy_output(filename: str = "test.png", data: bytes = None) -> ComfyOutput:
    """Create a test ComfyOutput."""
    if data is None:
        data = create_test_image()
    
    return ComfyOutput(
        data=data,
        filename=filename,
        subfolder="output",
        type="output"
    )


def create_workflow_with_duplicate_titles() -> Workflow:
    """Create a workflow with duplicate node titles for testing error conditions."""
    api_json = {
        "1": {
            "class_type": "KSampler",
            "_meta": {"title": "Duplicate Title"},
            "inputs": {"seed": 123}
        },
        "2": {
            "class_type": "KSampler", 
            "_meta": {"title": "Duplicate Title"},
            "inputs": {"seed": 456}
        }
    }
    gui_json = {"nodes": [], "links": []}
    return Workflow(api_json=api_json, gui_json=gui_json)


def create_workflow_with_multiple_same_class_type() -> Workflow:
    """Create a workflow with multiple nodes of the same class type."""
    api_json = {
        "1": {
            "class_type": "KSampler",
            "_meta": {"title": "Sampler 1"},
            "inputs": {"seed": 123}
        },
        "2": {
            "class_type": "KSampler", 
            "_meta": {"title": "Sampler 2"},
            "inputs": {"seed": 456}
        },
        "3": {
            "class_type": "KSampler",
            "_meta": {"title": "Sampler 1"},  # Duplicate title
            "inputs": {"seed": 789}
        }
    }
    gui_json = {"nodes": [], "links": []}
    return Workflow(api_json=api_json, gui_json=gui_json)


def safe_cleanup_file(file_path: str) -> None:
    """Safely clean up a temporary file, handling permission errors on Windows."""
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked
            pass


def with_temp_file(suffix: str = '.tmp', delete: bool = True):
    """Context manager for creating temporary files with automatic cleanup."""
    class TempFileContext:
        def __init__(self, suffix, delete):
            self.suffix = suffix
            self.delete = delete
            self.temp_file = None
            
        def __enter__(self):
            self.temp_file = tempfile.NamedTemporaryFile(suffix=self.suffix, delete=False)
            return self.temp_file.name
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.delete:
                safe_cleanup_file(self.temp_file.name)
    
    return TempFileContext(suffix, delete)
