"""
Tests for ComfyUIServer execution and queue functionality.
"""

import pytest
import threading
from unittest.mock import patch

from comfy_commander import Workflow, ComfyUIServer, ComfyOutput, ExecutionResult


class TestServerExecution:
    """Test ComfyUIServer execution and queue functionality."""

    def test_server_queue_method(self):
        """Test server.queue(workflow) returns prompt ID immediately."""
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
                 patch.object(ComfyUIServer, 'get_outputs', return_value=[ComfyOutput(data=b"fake_image", filename="test_output.png")]):
                
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
             patch.object(ComfyUIServer, 'get_outputs', return_value=[ComfyOutput(data=b"fake_image", filename="test_output.png")]):
            
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


class TestExecutionResult:
    """Test ExecutionResult creation and properties."""

    def test_execution_result_creation(self):
        """Test ExecutionResult creation and properties."""
        # Create test outputs
        output1 = ComfyOutput(data=b"fake_output_data_1", filename="test1.png")
        output2 = ComfyOutput(data=b"fake_output_data_2", filename="test2.mp4")
        
        # Create ExecutionResult
        result = ExecutionResult(
            prompt_id="test_prompt_123",
            media=[output1, output2],
            status="success"
        )
        
        # Verify properties
        assert result.prompt_id == "test_prompt_123"
        assert len(result.media) == 2
        assert result.media[0].filename == "test1.png"
        assert result.media[1].filename == "test2.mp4"
        assert result.media[0].is_image
        assert result.media[1].is_video
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
