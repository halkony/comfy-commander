import pytest
import asyncio
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from comfy_commander import Workflow, ComfyUIServer, ComfyImage, ExecutionResult
from helpers import (
    assert_api_param_updated,
    assert_gui_widget_updated,
    assert_connections_preserved,
    assert_gui_connections_preserved
)


class TestWorkflows:
    def test_workflow_node_editable_by_id(self, example_api_workflow_file_path):
        workflow = Workflow.from_file(example_api_workflow_file_path)
        workflow.node(id="31").param("seed").set(1234567890)
        assert workflow.node(id="31").param("seed").value == 1234567890

    def test_workflow_node_editable_by_title(self, example_api_workflow_file_path):
        workflow = Workflow.from_file(example_api_workflow_file_path)
        workflow.node(title="KSampler").param("seed").set(1234567890)
        assert workflow.node(title="KSampler").param("seed").value == 1234567890

    def test_workflow_node_editable_by_class_type(self, example_api_workflow_file_path):
        workflow = Workflow.from_file(example_api_workflow_file_path)
        workflow.node(class_type="KSampler").param("seed").set(1234567890)
        assert workflow.node(class_type="KSampler").param("seed").value == 1234567890

    def test_workflow_node_class_type_error_multiple_nodes(self, example_api_workflow_file_path):
        """Test that class_type throws an error when multiple nodes of the same type exist."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # This should raise a ValueError because there are multiple CLIPTextEncode nodes
        with pytest.raises(ValueError, match="Multiple nodes found with class_type 'CLIPTextEncode'"):
            workflow.node(class_type="CLIPTextEncode")

    def test_workflow_node_title_error_multiple_nodes(self):
        """Test that title throws an error when multiple nodes with the same title exist."""
        # Create a workflow with duplicate titles
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
        workflow = Workflow(api_json=api_json, gui_json=gui_json)
        
        # This should raise a ValueError because there are multiple nodes with the same title
        with pytest.raises(ValueError, match="Multiple nodes found with title 'Duplicate Title'"):
            workflow.node(title="Duplicate Title")

    def test_workflow_nodes_by_class_type(self, example_api_workflow_file_path):
        """Test that workflow.nodes() returns all nodes with the given class_type."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Get all CLIPTextEncode nodes
        clip_nodes = workflow.nodes(class_type="CLIPTextEncode")
        
        # Should return 2 nodes (positive and negative prompt encoders)
        assert len(clip_nodes) == 2
        
        # Verify they are all CLIPTextEncode nodes
        for node in clip_nodes:
            assert node.class_type == "CLIPTextEncode"
        
        # Verify we can access their properties
        for node in clip_nodes:
            assert hasattr(node, 'param')
            assert hasattr(node, 'class_type')

    def test_workflow_nodes_by_title(self, example_api_workflow_file_path):
        """Test that workflow.nodes() returns all nodes with the given title."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Get all nodes with the title "CLIP Text Encode (Positive Prompt)"
        positive_nodes = workflow.nodes(title="CLIP Text Encode (Positive Prompt)")
        
        # Should return exactly 1 node
        assert len(positive_nodes) == 1
        
        # Verify it has the correct title
        assert positive_nodes[0].title == "CLIP Text Encode (Positive Prompt)"
        assert positive_nodes[0].class_type == "CLIPTextEncode"

    def test_workflow_nodes_multiple_matches(self):
        """Test that workflow.nodes() returns multiple nodes when there are duplicates."""
        # Create a workflow with multiple nodes of the same class_type and title
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
        workflow = Workflow(api_json=api_json, gui_json=gui_json)
        
        # Test by class_type - should return all 3 KSampler nodes
        sampler_nodes = workflow.nodes(class_type="KSampler")
        assert len(sampler_nodes) == 3
        
        # Test by title - should return 2 nodes with "Sampler 1" title
        sampler1_nodes = workflow.nodes(title="Sampler 1")
        assert len(sampler1_nodes) == 2
        
        # Verify we can access properties of all returned nodes
        for node in sampler_nodes:
            assert node.class_type == "KSampler"
            assert hasattr(node, 'param')

    def test_workflow_nodes_no_matches(self, example_api_workflow_file_path):
        """Test that workflow.nodes() returns empty list when no nodes match."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Search for non-existent class_type
        non_existent_nodes = workflow.nodes(class_type="NonExistentNode")
        assert len(non_existent_nodes) == 0
        
        # Search for non-existent title
        non_existent_title_nodes = workflow.nodes(title="Non Existent Title")
        assert len(non_existent_title_nodes) == 0

    def test_workflow_nodes_error_no_parameters(self):
        """Test that workflow.nodes() raises error when no parameters are provided."""
        api_json = {"1": {"class_type": "KSampler", "inputs": {}}}
        gui_json = {"nodes": [], "links": []}
        workflow = Workflow(api_json=api_json, gui_json=gui_json)
        
        with pytest.raises(ValueError, match="Either 'title' or 'class_type' must be provided"):
            workflow.nodes()

    def test_workflow_nodes_editable_properties(self):
        """Test that nodes returned by workflow.nodes() are editable."""
        api_json = {
            "1": {
                "class_type": "KSampler",
                "_meta": {"title": "Test Sampler"},
                "inputs": {"seed": 123, "steps": 20}
            },
            "2": {
                "class_type": "KSampler",
                "_meta": {"title": "Test Sampler"},
                "inputs": {"seed": 456, "steps": 30}
            }
        }
        gui_json = {"nodes": [], "links": []}
        workflow = Workflow(api_json=api_json, gui_json=gui_json)
        
        # Get all nodes with the same title
        test_nodes = workflow.nodes(title="Test Sampler")
        assert len(test_nodes) == 2
        
        # Modify properties of each node
        test_nodes[0].param("seed").set(999)
        test_nodes[1].param("steps").set(50)
        
        # Verify the changes were applied
        assert test_nodes[0].param("seed").value == 999
        assert test_nodes[1].param("steps").value == 50

    def test_can_load_workflow_from_example_image(self, snapshot, example_image_file_path):
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
    
    def test_dual_workflow_synchronization_api_to_gui(self, example_image_file_path):
        """Test that changes to API JSON are synchronized to GUI JSON."""
        workflow = Workflow.from_image(example_image_file_path)
        
        # Get the KSampler node and change the seed
        node = workflow.node(id="31")
        new_seed = 999999999
        
        # Change the seed in API JSON
        node.param("seed").set(new_seed)
        
        # Verify API JSON was updated
        assert_api_param_updated(workflow, "31", "seed", new_seed)
        
        # Verify GUI JSON was synchronized (seed is at index 0 for KSampler)
        assert_gui_widget_updated(workflow, 31, 0, new_seed)
    
    def test_dual_workflow_synchronization_multiple_properties(self, example_image_file_path):
        """Test that multiple property changes are synchronized correctly."""
        workflow = Workflow.from_image(example_image_file_path)
        
        # Get the KSampler node and change multiple properties
        node = workflow.node(id="31")
        
        # Change multiple properties
        node.param("seed").set(111111111)
        node.param("steps").set(20)
        node.param("cfg").set(2.5)
        
        # Verify API JSON was updated
        assert_api_param_updated(workflow, "31", "seed", 111111111)
        assert_api_param_updated(workflow, "31", "steps", 20)
        assert_api_param_updated(workflow, "31", "cfg", 2.5)
        
        # Verify GUI JSON was synchronized (order: seed, randomize, steps, cfg at indices 0, 1, 2, 3)
        assert_gui_widget_updated(workflow, 31, 0, 111111111)  # seed
        assert_gui_widget_updated(workflow, 31, 2, 20)         # steps
        assert_gui_widget_updated(workflow, 31, 3, 2.5)        # cfg
    
    def test_dual_workflow_synchronization_text_property(self, example_image_file_path):
        """Test that text properties are synchronized correctly."""
        workflow = Workflow.from_image(example_image_file_path)
        
        # Get the CLIPTextEncode node and change the text
        node = workflow.node(id="6")
        new_text = "A beautiful landscape with mountains and rivers"
        
        # Change the text property
        node.param("text").set(new_text)
        
        # Verify API JSON was updated
        assert_api_param_updated(workflow, "6", "text", new_text)
        
        # Verify GUI JSON was synchronized (text is at index 0 for CLIPTextEncode)
        assert_gui_widget_updated(workflow, 6, 0, new_text)
    
    def test_dual_workflow_synchronization_preserves_connections(self, example_image_file_path):
        """Test that property changes don't affect node connections."""
        workflow = Workflow.from_image(example_image_file_path)
        
        # Get the KSampler node and change a property
        node = workflow.node(id="31")
        node.param("seed").set(555555555)
        
        # Verify that connections are preserved in API JSON
        expected_connections = ["model", "positive", "negative", "latent_image"]
        assert_connections_preserved(workflow, "31", expected_connections)
        
        # Verify that connections are preserved in GUI JSON
        assert_gui_connections_preserved(workflow, 31, 4, 1)  # 4 inputs, 1 output
    
    def test_dual_workflow_synchronization_node_by_name(self, example_image_file_path):
        """Test that synchronization works when accessing nodes by name."""
        workflow = Workflow.from_image(example_image_file_path)
        
        # Get the KSampler node by name and change a property
        node = workflow.node(name="KSampler")
        node.param("seed").set(777777777)
        
        # Verify both JSON formats were updated
        assert_api_param_updated(workflow, "31", "seed", 777777777)
        assert_gui_widget_updated(workflow, 31, 0, 777777777)

    def test_comfy_image_creation_and_save(self):
        """Test ComfyImage creation and save functionality."""
        # Create a simple test image
        from PIL import Image
        import io
        
        # Create a 100x100 red image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create ComfyImage
        comfy_image = ComfyImage(
            data=img_data,
            filename="test.png",
            subfolder="output",
            type="output"
        )
        
        # Test saving to file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            comfy_image.save(tmp_path)
            
            # Verify the file was created and contains the image
            assert os.path.exists(tmp_path)
            saved_image = Image.open(tmp_path)
            assert saved_image.size == (100, 100)
            assert saved_image.mode == 'RGB'
            saved_image.close()  # Close the image to release file handle
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_comfy_image_save_with_workflow_metadata(self):
        """Test ComfyImage save functionality with workflow metadata embedding."""
        # Create a simple test image
        from PIL import Image
        import io
        
        # Create a 100x100 red image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create a test workflow
        test_workflow = Workflow(
            api_json={"1": {"class_type": "TestNode", "inputs": {"test": "value"}}},
            gui_json={"nodes": [{"id": 1, "type": "TestNode"}]}
        )
        
        # Create ComfyImage with workflow reference
        comfy_image = ComfyImage(
            data=img_data,
            filename="test_with_workflow.png",
            subfolder="output",
            type="output"
        )
        comfy_image._workflow = test_workflow
        
        # Test saving to file with workflow metadata
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            comfy_image.save(tmp_path)
            
            # Verify the file was created
            assert os.path.exists(tmp_path)
            
            # Verify the image can be opened and has the correct properties
            saved_image = Image.open(tmp_path)
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
            
            saved_image.close()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_workflow_from_image_with_metadata(self):
        """Test loading a workflow from an image with embedded metadata."""
        # Create a simple test image
        from PIL import Image
        import io
        
        # Create a 100x100 red image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create a test workflow
        test_workflow = Workflow(
            api_json={"1": {"class_type": "TestNode", "inputs": {"test": "value"}}},
            gui_json={"nodes": [{"id": 1, "type": "TestNode"}]}
        )
        
        # Create ComfyImage with workflow reference
        comfy_image = ComfyImage(
            data=img_data,
            filename="test_workflow_roundtrip.png",
            subfolder="output",
            type="output"
        )
        comfy_image._workflow = test_workflow
        
        # Save the image with metadata
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            comfy_image.save(tmp_path)
            
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
        from PIL import Image
        import io
        
        # Create a 100x100 red image
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

    def test_comfy_image_from_base64(self):
        """Test ComfyImage creation from base64 data."""
        import base64
        
        # Create test image data
        from PIL import Image
        import io
        
        test_image = Image.new('RGB', (50, 50), color='blue')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Encode to base64
        base64_data = base64.b64encode(img_data).decode('utf-8')
        
        # Create ComfyImage from base64
        comfy_image = ComfyImage.from_base64(
            base64_data,
            filename="base64_test.png",
            subfolder="test",
            type="input"
        )
        
        # Verify properties
        assert comfy_image.filename == "base64_test.png"
        assert comfy_image.subfolder == "test"
        assert comfy_image.type == "input"
        assert len(comfy_image.data) > 0

    def test_execution_result_creation(self):
        """Test ExecutionResult creation and properties."""
        # Create test images
        image1 = ComfyImage(data=b"fake_image_data_1", filename="test1.png")
        image2 = ComfyImage(data=b"fake_image_data_2", filename="test2.png")
        
        # Create ExecutionResult
        result = ExecutionResult(
            prompt_id="test_prompt_123",
            media=[image1, image2],
            status="success"
        )
        
        # Verify properties
        assert result.prompt_id == "test_prompt_123"
        assert len(result.media) == 2
        assert result.media[0].filename == "test1.png"
        assert result.media[1].filename == "test2.png"
        assert result.status == "success"
        assert result.error_message is None

    def test_execution_result_with_error(self):
        """Test ExecutionResult with error status."""
        result = ExecutionResult(
            prompt_id="failed_prompt_456",
            media=[],
            status="error",
            error_message="Test error message"
        )
        
        assert result.prompt_id == "failed_prompt_456"
        assert len(result.media) == 0
        assert result.status == "error"
        assert result.error_message == "Test error message"

    def test_server_queue_method(self):
        """Test server.queue(workflow) returns prompt ID immediately."""
        from unittest.mock import patch
        
        # Create a real ComfyUIServer instance
        server = ComfyUIServer("http://localhost:8188")
        
        # Mock the _send_workflow_to_server method at class level
        with patch.object(ComfyUIServer, '_send_workflow_to_server', return_value="test_prompt_123"):
            # Create workflow
            api_json = {"1": {"class_type": "KSampler", "inputs": {"seed": 123}}}
            gui_json = {"nodes": [], "links": []}
            workflow = Workflow(api_json=api_json, gui_json=gui_json)
            
            # Queue the workflow
            result = server.queue(workflow)
            
            # Should return just the prompt ID
            assert result == "test_prompt_123"
    
    def test_server_execute_sync_mode(self):
        """Test server.execute(workflow) in synchronous mode waits for completion."""
        from unittest.mock import patch
        import threading
        
        def run_in_thread():
            # Create a real ComfyUIServer instance
            server = ComfyUIServer("http://localhost:8188")
            
            # Mock the async methods
            mock_execution_data = {
                "status": {"status_str": "success"},
                "outputs": {}
            }
            
            # Create a coroutine for the async method
            async def mock_wait_for_completion(*args, **kwargs):
                return mock_execution_data
            
            # Mock the methods at class level
            with patch.object(ComfyUIServer, '_send_workflow_to_server', return_value="test_prompt_123"), \
                 patch.object(ComfyUIServer, 'wait_for_completion', side_effect=mock_wait_for_completion), \
                 patch.object(ComfyUIServer, 'get_output_images', return_value=[ComfyImage(data=b"fake_image", filename="test_output.png")]):
                
                # Create workflow
                api_json = {"1": {"class_type": "KSampler", "inputs": {"seed": 123}}}
                gui_json = {"nodes": [], "links": []}
                workflow = Workflow(api_json=api_json, gui_json=gui_json)
                
                # Execute in sync mode (should wait for completion)
                result = server.execute(workflow)
                
                # Should return ExecutionResult
                assert isinstance(result, ExecutionResult)
                assert result.prompt_id == "test_prompt_123"
                assert result.status == "success"
                assert len(result.media) == 1
        
        # Run in a separate thread to avoid async context
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

    @pytest.mark.asyncio
    async def test_server_execute_async_mode(self):
        """Test server.execute(workflow) in asynchronous mode."""
        from unittest.mock import patch
        
        # Create a real ComfyUIServer instance
        server = ComfyUIServer("http://localhost:8188")
        
        # Mock the async methods
        mock_execution_data = {
            "status": {"status_str": "success"},
            "outputs": {
                "31": {
                    "images": [
                        {
                            "filename": "test_output.png",
                            "subfolder": "output",
                            "type": "output"
                        }
                    ]
                }
            }
        }
        
        # Create a coroutine for the async method
        async def mock_wait_for_completion(*args, **kwargs):
            return mock_execution_data
        
        # Create workflow
        api_json = {"1": {"class_type": "KSampler", "inputs": {"seed": 123}}}
        gui_json = {"nodes": [], "links": []}
        workflow = Workflow(api_json=api_json, gui_json=gui_json)
        
        # Mock the methods at class level
        with patch.object(ComfyUIServer, '_send_workflow_to_server', return_value="test_prompt_123"), \
             patch.object(ComfyUIServer, 'wait_for_completion', side_effect=mock_wait_for_completion), \
             patch.object(ComfyUIServer, 'get_output_images', return_value=[ComfyImage(data=b"fake_image", filename="test_output.png")]):
            
            # Execute in async mode
            result = await server.execute_async(workflow)
            
            # Should return ExecutionResult
            assert isinstance(result, ExecutionResult)
            assert result.prompt_id == "test_prompt_123"
            assert result.status == "success"
            assert len(result.media) == 1
            assert result.media[0].filename == "test_output.png"

    @pytest.mark.asyncio
    async def test_server_execute_async_with_error(self):
        """Test server.execute(workflow) in async mode with execution error."""
        from unittest.mock import patch
        
        # Create a real ComfyUIServer instance
        server = ComfyUIServer("http://localhost:8188")
        
        # Create workflow
        api_json = {"1": {"class_type": "KSampler", "inputs": {"seed": 123}}}
        gui_json = {"nodes": [], "links": []}
        workflow = Workflow(api_json=api_json, gui_json=gui_json)
        
        # Mock the methods to simulate an error
        with patch.object(ComfyUIServer, '_send_workflow_to_server', return_value="test_prompt_456"), \
             patch.object(ComfyUIServer, 'wait_for_completion', side_effect=RuntimeError("Execution failed")):
            
            # Execute in async mode
            result = await server.execute_async(workflow)
            
            # Should return ExecutionResult with error
            assert isinstance(result, ExecutionResult)
            assert result.prompt_id == "test_prompt_456"
            assert result.status == "error"
            assert "Execution failed" in result.error_message
            assert len(result.media) == 0


class TestComfyImageNodeAttribute:
    """Test ComfyImage node attribute functionality."""
    
    def test_comfyimage_creation_with_node(self, example_api_workflow_file_path):
        """Test creating ComfyImage with node reference."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        node = workflow.node(id="31")  # KSampler node
        
        image_data = b"fake_image_data"
        image = ComfyImage(
            data=image_data,
            filename="test.png",
            subfolder="output",
            type="output",
            node=node
        )
        
        assert image.node is not None
        assert image.node.id == "31"
        assert image.node.class_type == "KSampler"
        assert image.node.workflow == workflow
    
    def test_comfyimage_creation_without_node(self):
        """Test creating ComfyImage without node reference."""
        image_data = b"fake_image_data"
        image = ComfyImage(
            data=image_data,
            filename="test.png",
            subfolder="output",
            type="output"
        )
        
        assert image.node is None
    
    def test_comfyimage_from_base64_with_node(self, example_api_workflow_file_path):
        """Test creating ComfyImage from base64 with node reference."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        node = workflow.node(id="31")  # KSampler node
        
        import base64
        image_data = b"fake_image_data"
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        image = ComfyImage.from_base64(
            base64_data=base64_data,
            filename="test.png",
            subfolder="output",
            type="output",
            node=node
        )
        
        assert image.node is not None
        assert image.node.id == "31"
        assert image.node.class_type == "KSampler"
    
    def test_comfyimage_from_base64_without_node(self):
        """Test creating ComfyImage from base64 without node reference."""
        import base64
        image_data = b"fake_image_data"
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        image = ComfyImage.from_base64(
            base64_data=base64_data,
            filename="test.png",
            subfolder="output",
            type="output"
        )
        
        assert image.node is None
    
    @patch('requests.get')
    def test_get_output_images_sets_node_reference(self, mock_get, example_api_workflow_file_path):
        """Test that get_output_images sets node reference correctly."""
        # Mock the image data response
        mock_response = Mock()
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock the history response
        mock_history = {
            "test_prompt_123": {
                "outputs": {
                    "31": {  # KSampler node
                        "images": [
                            {
                                "filename": "test_output.png",
                                "subfolder": "output",
                                "type": "output"
                            }
                        ]
                    }
                }
            }
        }
        
        server = ComfyUIServer()
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Mock the get_history method using patch
        with patch.object(ComfyUIServer, 'get_history', return_value=mock_history):
            images = server.get_output_images("test_prompt_123", workflow)
        
        assert len(images) == 1
        image = images[0]
        
        # Check that node reference is set correctly
        assert image.node is not None
        assert image.node.id == "31"
        assert image.node.class_type == "KSampler"
        assert image.node.workflow == workflow
    
    @patch('requests.get')
    def test_get_output_images_without_workflow(self, mock_get):
        """Test that get_output_images works without workflow (node should be None)."""
        # Mock the image data response
        mock_response = Mock()
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock the history response
        mock_history = {
            "test_prompt_123": {
                "outputs": {
                    "31": {  # KSampler node
                        "images": [
                            {
                                "filename": "test_output.png",
                                "subfolder": "output",
                                "type": "output"
                            }
                        ]
                    }
                }
            }
        }
        
        server = ComfyUIServer()
        
        # Mock the get_history method using patch
        with patch.object(ComfyUIServer, 'get_history', return_value=mock_history):
            images = server.get_output_images("test_prompt_123", None)
        
        assert len(images) == 1
        image = images[0]
        
        # Check that node reference is None when no workflow provided
        assert image.node is None
    
    @patch('requests.get')
    def test_get_output_images_node_not_in_workflow(self, mock_get, example_api_workflow_file_path):
        """Test that get_output_images handles case where node is not in workflow."""
        # Mock the image data response
        mock_response = Mock()
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock the history response with a node ID that doesn't exist in workflow
        mock_history = {
            "test_prompt_123": {
                "outputs": {
                    "999": {  # Non-existent node
                        "images": [
                            {
                                "filename": "test_output.png",
                                "subfolder": "output",
                                "type": "output"
                            }
                        ]
                    }
                }
            }
        }
        
        server = ComfyUIServer()
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Mock the get_history method using patch
        with patch.object(ComfyUIServer, 'get_history', return_value=mock_history):
            images = server.get_output_images("test_prompt_123", workflow)
        
        assert len(images) == 1
        image = images[0]
        
        # Check that a basic Node object is created for non-existent node
        assert image.node is not None
        assert image.node.id == "999"
        assert image.node.workflow == workflow
        # The class_type should be empty since it's not in the workflow
        assert image.node.class_type == ""
    
    def test_comfyimage_node_access_properties(self, example_api_workflow_file_path):
        """Test accessing node properties through ComfyImage."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        node = workflow.node(id="31")  # KSampler node
        
        image_data = b"fake_image_data"
        image = ComfyImage(
            data=image_data,
            filename="test.png",
            subfolder="output",
            type="output",
            node=node
        )
        
        # Test accessing node properties through the image
        assert image.node.class_type == "KSampler"
        assert image.node.id == "31"
        
        # Test accessing node parameters
        seed_value = image.node.param("seed").value
        assert seed_value is not None
        
        # Test modifying node parameters through the image
        image.node.param("seed").set(999999)
        assert image.node.param("seed").value == 999999
        
        # Verify the change is reflected in the workflow
        assert workflow.node(id="31").param("seed").value == 999999


class TestMediaCollection:
    """Test the MediaCollection class functionality."""
    
    def test_media_collection_iteration(self, example_api_workflow_file_path):
        """Test that MediaCollection can be iterated over like a list."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Create some test images with nodes
        node1 = workflow.node(id="31")  # KSampler node
        node2 = workflow.node(id="6")   # CLIPTextEncode node
        
        image1 = ComfyImage(
            data=b"fake_image_data_1",
            filename="test1.png",
            node=node1
        )
        image2 = ComfyImage(
            data=b"fake_image_data_2", 
            filename="test2.png",
            node=node2
        )
        
        # Create MediaCollection and add images
        from comfy_commander import MediaCollection
        media = MediaCollection()
        media.append(image1)
        media.append(image2)
        
        # Test iteration
        images_list = list(media)
        assert len(images_list) == 2
        assert images_list[0] == image1
        assert images_list[1] == image2
        
        # Test len
        assert len(media) == 2
        
        # Test indexing
        assert media[0] == image1
        assert media[1] == image2
    
    def test_media_collection_find_by_title_success(self, example_api_workflow_file_path):
        """Test finding an image by node title successfully."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Create test images with nodes that have titles
        node1 = workflow.node(id="31")  # KSampler node with title "KSampler"
        node2 = workflow.node(id="6")   # CLIPTextEncode node
        
        image1 = ComfyImage(
            data=b"fake_image_data_1",
            filename="test1.png",
            node=node1
        )
        image2 = ComfyImage(
            data=b"fake_image_data_2",
            filename="test2.png", 
            node=node2
        )
        
        from comfy_commander import MediaCollection
        media = MediaCollection()
        media.append(image1)
        media.append(image2)
        
        # Test finding by title
        found_image = media.find_by_title("KSampler")
        assert found_image == image1
        assert found_image.node.title == "KSampler"
    
    def test_media_collection_find_by_title_no_match(self, example_api_workflow_file_path):
        """Test that find_by_title raises KeyError when no match is found."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        node = workflow.node(id="31")  # KSampler node
        image = ComfyImage(
            data=b"fake_image_data",
            filename="test.png",
            node=node
        )
        
        from comfy_commander import MediaCollection
        media = MediaCollection()
        media.append(image)
        
        # Test that KeyError is raised for non-existent title
        with pytest.raises(KeyError, match="No image found with node title 'NonExistentTitle'"):
            media.find_by_title("NonExistentTitle")
    
    def test_media_collection_find_by_title_multiple_matches(self):
        """Test that find_by_title raises ValueError when multiple matches are found."""
        # Create a mock workflow with duplicate titles
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
        
        workflow = Workflow(api_json=api_json, gui_json=None)
        
        # Create images with nodes that have the same title
        node1 = workflow.node(id="1")
        node2 = workflow.node(id="2")
        
        image1 = ComfyImage(
            data=b"fake_image_data_1",
            filename="test1.png",
            node=node1
        )
        image2 = ComfyImage(
            data=b"fake_image_data_2",
            filename="test2.png",
            node=node2
        )
        
        from comfy_commander import MediaCollection
        media = MediaCollection()
        media.append(image1)
        media.append(image2)
        
        # Test that ValueError is raised for multiple matches
        with pytest.raises(ValueError, match="Multiple images found with node title 'Duplicate Title': 2 matches"):
            media.find_by_title("Duplicate Title")
    
    def test_media_collection_find_by_title_no_node(self):
        """Test that find_by_title handles images without nodes."""
        from comfy_commander import MediaCollection, ComfyImage
        
        # Create image without a node
        image = ComfyImage(
            data=b"fake_image_data",
            filename="test.png",
            node=None
        )
        
        media = MediaCollection()
        media.append(image)
        
        # Test that KeyError is raised when no node is present
        with pytest.raises(KeyError, match="No image found with node title 'SomeTitle'"):
            media.find_by_title("SomeTitle")
    
    def test_media_collection_extend(self):
        """Test extending MediaCollection with multiple images."""
        from comfy_commander import MediaCollection, ComfyImage
        
        image1 = ComfyImage(data=b"data1", filename="test1.png")
        image2 = ComfyImage(data=b"data2", filename="test2.png")
        image3 = ComfyImage(data=b"data3", filename="test3.png")
        
        media = MediaCollection()
        media.append(image1)
        media.extend([image2, image3])
        
        assert len(media) == 3
        assert media[0] == image1
        assert media[1] == image2
        assert media[2] == image3
    
    def test_media_collection_repr(self):
        """Test MediaCollection string representation."""
        from comfy_commander import MediaCollection, ComfyImage
        
        media = MediaCollection()
        assert repr(media) == "MediaCollection(0 images)"
        
        image = ComfyImage(data=b"data", filename="test.png")
        media.append(image)
        assert repr(media) == "MediaCollection(1 images)"
        
        media.append(image)
        assert repr(media) == "MediaCollection(2 images)"
    
    def test_execution_result_with_media_collection(self, example_api_workflow_file_path):
        """Test that ExecutionResult properly uses MediaCollection."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        node = workflow.node(id="31")
        image = ComfyImage(
            data=b"fake_image_data",
            filename="test.png",
            node=node
        )
        
        from comfy_commander import MediaCollection, ExecutionResult
        
        media = MediaCollection()
        media.append(image)
        
        result = ExecutionResult(
            prompt_id="test_123",
            media=media,
            status="success"
        )
        
        # Test that we can iterate over result.media
        images_list = list(result.media)
        assert len(images_list) == 1
        assert images_list[0] == image
        
        # Test that we can find by title
        found_image = result.media.find_by_title("KSampler")
        assert found_image == image
        
        # Test that result.media is a MediaCollection
        assert isinstance(result.media, MediaCollection)