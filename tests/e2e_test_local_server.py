"""
Tests for local ComfyUI server functionality.
"""

import json
import pytest
import requests
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from comfy_commander.core import ComfyUIServer, Workflow, ExecutionResult, ComfyOutput, MediaCollection

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
            "tests/fixtures/flux_dev_checkpoint_example_standard.json"
        )
        
        # Verify workflow loaded
        assert isinstance(workflow, Workflow)
        assert workflow.gui_json is not None  # Standard workflow should have GUI data
        assert workflow.api_json is None  # API data should be None until conversion
        
        # Queue the workflow (conversion will happen here)
        prompt_id = server.queue(workflow, "comfy-commander-test")
        
        # Verify execution started
        assert isinstance(prompt_id, str)
        assert len(prompt_id) > 0
        
        # Check queue status
        queue_status = server.get_queue_status()
        assert "queue_running" in queue_status
        assert "queue_pending" in queue_status
    
    def test_workflow_parameter_modification_after_conversion(self, server):
        """Test modifying workflow parameters after conversion."""
        # Load and convert standard workflow
        workflow = Workflow.from_file(
            "tests/fixtures/flux_dev_checkpoint_example_standard.json"
        )
        
        # First, ensure the workflow is converted to API format
        workflow.ensure_api_format(server)
        
        # Now modify parameters
        sampler_node = workflow.node(id="31")
        original_seed = sampler_node.param("seed").value
        new_seed = 1234567890
        
        sampler_node.param("seed").set(new_seed)
        
        # Verify the change
        assert sampler_node.param("seed").value == new_seed
        # After conversion, api_json should be populated
        assert workflow.api_json is not None
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

    @pytest.mark.asyncio
    async def test_workflow_execute_async_e2e(self, server):
        """Test the new async workflow execution API end-to-end."""
        # Skip if server is not available
        if not server.is_available():
            pytest.skip("ComfyUI server is not available")
        
        # Load a simple workflow that generates an image
        workflow_path = "tests/fixtures/flux_dev_checkpoint_example_api.json"
        
        if not os.path.exists(workflow_path):
            pytest.skip(f"Test workflow file not found: {workflow_path}")
        
        # Load the workflow
        workflow = Workflow.from_file(workflow_path)
        
        # Modify the workflow to use a simple seed for faster execution
        try:
            # Try to find and modify a KSampler node
            sampler_nodes = workflow.nodes(class_type="KSampler")
            if sampler_nodes:
                sampler_nodes[0].param("seed").set(12345)
                sampler_nodes[0].param("steps").set(1)  # Minimal steps for faster execution
        except Exception:
            # If we can't modify the workflow, that's okay for this test
            pass
        
        # Execute the workflow asynchronously
        result = await server.execute_async(workflow, timeout=60.0)  # 60 second timeout
        
        # Verify the result
        assert isinstance(result, ExecutionResult)
        assert result.prompt_id is not None
        assert len(result.prompt_id) > 0
        
        # The status should be either success or error
        assert result.status in ["success", "error"]
        
        if result.status == "success":
            # If successful, we should have some media
            assert isinstance(result.media, MediaCollection)
            
            # This workflow should produce images (it has a SaveImage node)
            # So we expect at least one image in the result
            assert len(result.media) > 0, "Workflow should produce at least one image but result.media is empty"
            
            # Verify that the images are valid ComfyOutput objects
            for i, output in enumerate(result.media):
                assert hasattr(output, 'data'), f"Output {i} should have data attribute"
                assert hasattr(output, 'filename'), f"Output {i} should have filename attribute"
                assert len(output.data) > 0, f"Output {i} should have non-empty data"
                assert output.filename, f"Output {i} should have a filename"
        else:
            # If there was an error, we should have an error message
            assert result.error_message is not None
            assert len(result.error_message) > 0

    def test_workflow_execute_sync_e2e(self, server):
        """Test the synchronous workflow execution API end-to-end."""
        # Skip if server is not available
        if not server.is_available():
            pytest.skip("ComfyUI server is not available")
        
        # Load a simple workflow
        workflow_path = "tests/fixtures/flux_dev_checkpoint_example_api.json"
        
        if not os.path.exists(workflow_path):
            pytest.skip(f"Test workflow file not found: {workflow_path}")
        
        # Load the workflow
        workflow = Workflow.from_file(workflow_path)
        
        # Queue the workflow (should return just prompt ID)
        prompt_id = server.queue(workflow)
        
        # Verify the result
        assert isinstance(prompt_id, str)
        assert len(prompt_id) > 0
        
        # We should be able to get history for this prompt
        history = server.get_history(prompt_id)
        assert isinstance(history, dict)

    def test_workflow_execute_sync_with_wait_e2e(self, server):
        """Test synchronous workflow execution with waiting for completion."""
        # Skip if server is not available
        if not server.is_available():
            pytest.skip("ComfyUI server is not available")
        
        # Load a simple workflow
        workflow_path = "tests/fixtures/flux_dev_checkpoint_example_api.json"
        
        if not os.path.exists(workflow_path):
            pytest.skip(f"Test workflow file not found: {workflow_path}")
        
        # Load the workflow
        workflow = Workflow.from_file(workflow_path)
        
        # Modify the workflow to use minimal steps for faster execution
        try:
            sampler_nodes = workflow.nodes(class_type="KSampler")
            if sampler_nodes:
                sampler_nodes[0].param("seed").set(12345)
                sampler_nodes[0].param("steps").set(1)  # Minimal steps for faster execution
        except Exception:
            # If we can't modify the workflow, that's okay for this test
            pass
        
        # Execute the workflow synchronously and wait for completion
        result = server.execute(workflow, timeout=60.0)
        
        # Verify the result
        assert isinstance(result, ExecutionResult)
        assert result.prompt_id is not None
        assert len(result.prompt_id) > 0
        
        # The status should be either success or error
        assert result.status in ["success", "error"]
        
        if result.status == "success":
            # If successful, we should have some media
            assert isinstance(result.media, MediaCollection)
            
            # This workflow should produce images (it has a SaveImage node)
            # So we expect at least one image in the result
            assert len(result.media) > 0, "Workflow should produce at least one image but result.media is empty"
            
            # Verify that the images are valid ComfyOutput objects
            for i, output in enumerate(result.media):
                assert hasattr(output, 'data'), f"Output {i} should have data attribute"
                assert hasattr(output, 'filename'), f"Output {i} should have filename attribute"
                assert len(output.data) > 0, f"Output {i} should have non-empty data"
                assert output.filename, f"Output {i} should have a filename"
        else:
            # If there was an error, we should have an error message
            assert result.error_message is not None
            assert len(result.error_message) > 0

    @pytest.mark.asyncio
    async def test_image_save_functionality_e2e(self, server):
        """Test that generated images can be saved to files."""
        # Skip if server is not available
        if not server.is_available():
            pytest.skip("ComfyUI server is not available")
        
        # Load a simple workflow
        workflow_path = "tests/fixtures/flux_dev_checkpoint_example_api.json"
        
        if not os.path.exists(workflow_path):
            pytest.skip(f"Test workflow file not found: {workflow_path}")
        
        # Load the workflow
        workflow = Workflow.from_file(workflow_path)
        
        # Execute the workflow asynchronously
        result = await server.execute_async(workflow, timeout=60.0)
        assert len(result.media) > 0, "No images produced"
        
        # If we have images, test saving them
        if result.status == "success" and result.media:
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, output in enumerate(result.media):
                    # Test saving the output
                    output_path = os.path.join(temp_dir, f"test_output_{i}.png")
                    output.save(output_path)
                    
                    # Verify the file was created
                    assert os.path.exists(output_path)
                    assert os.path.getsize(output_path) > 0
                    
                    # Verify it's a valid image file
                    from PIL import Image
                    try:
                        saved_image = Image.open(output_path)
                        assert saved_image.size[0] > 0
                        assert saved_image.size[1] > 0
                        saved_image.close()
                    except Exception as e:
                        pytest.fail(f"Saved image is not valid: {e}")

    def test_comfy_image_creation_e2e(self):
        """Test ComfyOutput creation and manipulation without server."""
        # Create a simple test image
        from PIL import Image
        import io
        
        # Create a 100x100 test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create ComfyOutput
        comfy_output = ComfyOutput(
            data=img_data,
            filename="test_e2e.png",
            subfolder="output",
            type="output"
        )
        
        # Test saving to file
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_e2e_output.png")
            comfy_output.save(output_path)
            
            # Verify the file was created and is valid
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Verify it's a valid image
            saved_image = Image.open(output_path)
            assert saved_image.size == (100, 100)
            assert saved_image.mode == 'RGB'
            saved_image.close()

    def test_comfy_image_metadata_embedding_e2e(self):
        """Test ComfyOutput metadata embedding functionality end-to-end."""
        # Create a simple test image
        from PIL import Image
        import io
        
        # Create a 100x100 test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create a test workflow
        test_workflow = Workflow(
            api_json={"1": {"class_type": "TestNode", "inputs": {"test": "e2e_value"}}},
            gui_json={"nodes": [{"id": 1, "type": "TestNode"}]}
        )
        
        # Create ComfyOutput with workflow reference
        comfy_output = ComfyOutput(
            data=img_data,
            filename="test_e2e_metadata.png",
            subfolder="output",
            type="output"
        )
        comfy_output._workflow = test_workflow
        
        # Test saving to file with metadata
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_e2e_metadata_output.png")
            comfy_output.save(output_path)
            
            # Verify the file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Verify the image can be opened and has the correct properties
            saved_image = Image.open(output_path)
            assert saved_image.size == (100, 100)
            assert saved_image.mode == 'RGB'
            
            # Verify workflow metadata is embedded in image.info
            assert 'prompt' in saved_image.info
            assert 'workflow' in saved_image.info
            
            # Parse the metadata
            prompt_data = json.loads(saved_image.info['prompt'])
            workflow_data = json.loads(saved_image.info['workflow'])
            
            # Verify the metadata structure
            assert prompt_data == test_workflow.api_json
            assert workflow_data == test_workflow.gui_json
            
            # Test loading the workflow back from the image
            loaded_workflow = Workflow.from_image(output_path)
            assert loaded_workflow.api_json == test_workflow.api_json
            assert loaded_workflow.gui_json == test_workflow.gui_json
            
            saved_image.close()
