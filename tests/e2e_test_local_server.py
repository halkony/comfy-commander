"""
Tests for local ComfyUI server functionality.
"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock

from comfy_commander.core import ComfyUIServer, Workflow

# E2E Tests - These require a running ComfyUI instance with the workflow converter extension
class TestComfyUIServerE2E:
    """End-to-end tests for ComfyUIServer functionality.
    
    These tests require a running ComfyUI instance with the workflow converter extension.
    Run with: pytest tests/e2e_test_local_server.py
    """
    
    @pytest.fixture
    def server(self):
        """Create a ComfyUIServer instance for testing."""
        return ComfyUIServer()
    
    def test_server_availability(self, server):
        """Test that the ComfyUI server is available."""
        assert server.is_available(), "ComfyUI server is not running or not accessible"
    
    def test_convert_standard_workflow_to_api(self, server):
        """Test converting a standard workflow to API format using the /workflow/convert endpoint."""
        # Load the standard workflow fixture
        standard_workflow_path = "tests/fixtures/flux_dev_checkpoint_example_standard.json"
        
        with open(standard_workflow_path, 'r', encoding='utf-8') as f:
            standard_workflow = json.load(f)
        
        # Convert to API format
        api_workflow = server.convert_workflow(standard_workflow)
        
        # Verify the conversion result
        assert isinstance(api_workflow, dict), "API workflow should be a dictionary"
        assert len(api_workflow) > 0, "API workflow should not be empty"
        
        # Check that all nodes have the expected structure
        for node_id, node_data in api_workflow.items():
            assert "class_type" in node_data, f"Node {node_id} should have class_type"
            assert "inputs" in node_data, f"Node {node_id} should have inputs"
        
        # Verify specific nodes exist (based on the fixture)
        assert "6" in api_workflow, "CLIPTextEncode node should exist"
        assert "31" in api_workflow, "KSampler node should exist"
        assert "30" in api_workflow, "CheckpointLoaderSimple node should exist"
        
        # Verify specific node properties
        clip_node = api_workflow["6"]
        assert clip_node["class_type"] == "CLIPTextEncode"
        assert "text" in clip_node["inputs"]
        assert "clip" in clip_node["inputs"]
        
        sampler_node = api_workflow["31"]
        assert sampler_node["class_type"] == "KSampler"
        assert "seed" in sampler_node["inputs"]
        assert "steps" in sampler_node["inputs"]
        assert "cfg" in sampler_node["inputs"]
        
        # Verify the text content matches
        expected_text = "cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open placing a fancy black forest cake with candles on top of a dinner table of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere there are paintings on the walls"
        assert clip_node["inputs"]["text"] == expected_text
    
    def test_standard_workflow_conversion_and_execution(self, server):
        """Test loading a standard workflow, converting it, and executing it."""
        # Load standard workflow with server for automatic conversion
        workflow = Workflow.from_file(
            "tests/fixtures/flux_dev_checkpoint_example_standard.json",
            server
        )
        
        # Verify conversion
        assert isinstance(workflow, Workflow)
        assert len(workflow.api_json) > 0
        
        # Modify a parameter to make execution faster
        sampler_node = workflow.node(id="31")
        sampler_node.param("steps").set(1)  # Reduce steps for faster execution
        
        # Execute the workflow
        prompt_id = workflow.execute(server, "comfy-commander-test")
        
        # Verify execution started
        assert isinstance(prompt_id, str)
        assert len(prompt_id) > 0
        
        # Check queue status
        queue_status = server.get_queue_status()
        assert "queue_running" in queue_status
        assert "queue_pending" in queue_status
    
    def test_standard_workflow_direct_execution(self, server):
        """Test direct execution of a standard workflow."""
        # Load standard workflow with server
        workflow = Workflow.from_file(
            "tests/fixtures/flux_dev_checkpoint_example_standard.json",
            server
        )
        
        # Execute directly (converts and executes in one step)
        prompt_id = workflow.execute(server, "comfy-commander-direct-test")
        
        # Verify execution started
        assert isinstance(prompt_id, str)
        assert len(prompt_id) > 0
    
    def test_workflow_parameter_modification_after_conversion(self, server):
        """Test modifying workflow parameters after conversion."""
        # Load and convert standard workflow
        workflow = Workflow.from_file(
            "tests/fixtures/flux_dev_checkpoint_example_standard.json",
            server
        )
        
        # Modify parameters
        sampler_node = workflow.node(id="31")
        original_seed = sampler_node.param("seed").value
        new_seed = 1234567890
        
        sampler_node.param("seed").set(new_seed)
        
        # Verify the change
        assert sampler_node.param("seed").value == new_seed
        assert workflow.api_json["31"]["inputs"]["seed"] == new_seed
        
        # Verify the change is reflected in GUI JSON as well
        gui_sampler_node = None
        for node in workflow.gui_json["nodes"]:
            if str(node["id"]) == "31":
                gui_sampler_node = node
                break
        
        assert gui_sampler_node is not None
        # The seed should be at index 0 in widgets_values for KSampler
        assert gui_sampler_node["widgets_values"][0] == new_seed
    
    def test_queue_and_history_operations(self, server):
        """Test queue status and history operations."""
        # Get initial queue status
        queue_status = server.get_queue_status()
        assert isinstance(queue_status, dict)
        assert "queue_running" in queue_status
        assert "queue_pending" in queue_status
        
        # Get history
        history = server.get_history()
        assert isinstance(history, dict)
        
        # If there's history, test getting specific prompt history
        if history:
            prompt_id = list(history.keys())[0]
            specific_history = server.get_history(prompt_id)
            assert isinstance(specific_history, dict)
            assert prompt_id in specific_history
