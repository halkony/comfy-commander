"""
Test helpers for Comfy Commander tests.
"""

from typing import Dict, Any, Optional

from comfy_commander import Workflow


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
