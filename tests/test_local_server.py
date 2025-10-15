"""
Unit tests for local ComfyUI server functionality.
"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock

from comfy_commander.core import ComfyUIServer, Workflow


class TestComfyUIServer:
    """Test ComfyUIServer functionality."""
    
    def test_server_initialization(self):
        """Test server initialization with default values."""
        server = ComfyUIServer()
        assert server.base_url == "http://localhost:8188"
        assert server.timeout is None
    
    def test_server_initialization_custom_url(self):
        """Test server initialization with custom URL."""
        server = ComfyUIServer(base_url="http://192.168.1.100:8188", timeout=60)
        assert server.base_url == "http://192.168.1.100:8188"
        assert server.timeout == 60
    
    def test_server_url_normalization(self):
        """Test that trailing slashes are removed from base_url."""
        server = ComfyUIServer(base_url="http://localhost:8188/")
        assert server.base_url == "http://localhost:8188"
    
    @patch('requests.get')
    def test_is_available_success(self, mock_get):
        """Test server availability check when server is available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        server = ComfyUIServer()
        assert server.is_available() is True
        mock_get.assert_called_once_with("http://localhost:8188/system_stats", timeout=5)
    
    @patch('requests.get')
    def test_is_available_failure(self, mock_get):
        """Test server availability check when server is not available."""
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        server = ComfyUIServer()
        assert server.is_available() is False
    
    @patch('requests.get')
    def test_is_available_http_error(self, mock_get):
        """Test server availability check when server returns HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        server = ComfyUIServer()
        assert server.is_available() is False
    
    @patch('requests.post')
    def test_convert_workflow_success(self, mock_post):
        """Test successful workflow conversion."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
        mock_post.return_value = mock_response
        
        server = ComfyUIServer()
        workflow_data = {"nodes": [], "links": []}
        
        result = server.convert_workflow(workflow_data)
        
        assert result == {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
        mock_post.assert_called_once_with(
            "http://localhost:8188/workflow/convert",
            json=workflow_data,
            timeout=None
        )
    
    @patch('requests.post')
    def test_convert_workflow_failure(self, mock_post):
        """Test workflow conversion failure."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.HTTPError("Bad Request")
        mock_post.return_value = mock_response
        
        server = ComfyUIServer()
        workflow_data = {"invalid": "data"}
        
        with pytest.raises(requests.HTTPError):
            server.convert_workflow(workflow_data)
    
    @patch('requests.post')
    def test_send_workflow_to_server_success(self, mock_post):
        """Test successful workflow execution."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prompt_id": "test-prompt-123"}
        mock_post.return_value = mock_response
        
        server = ComfyUIServer()
        api_workflow = {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
        
        result = server._send_workflow_to_server(api_workflow, "test-client")
        
        assert result == "test-prompt-123"
        mock_post.assert_called_once_with(
            "http://localhost:8188/prompt",
            json={"prompt": api_workflow, "client_id": "test-client"},
            timeout=None
        )
    
    @patch('requests.get')
    def test_get_queue_status_success(self, mock_get):
        """Test successful queue status retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"queue_running": [], "queue_pending": []}
        mock_get.return_value = mock_response
        
        server = ComfyUIServer()
        result = server.get_queue_status()
        
        assert result == {"queue_running": [], "queue_pending": []}
        mock_get.assert_called_once_with("http://localhost:8188/queue", timeout=None)
    
    @patch('requests.get')
    def test_get_history_success(self, mock_get):
        """Test successful history retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test-prompt-123": {"status": "success"}}
        mock_get.return_value = mock_response
        
        server = ComfyUIServer()
        result = server.get_history("test-prompt-123")
        
        assert result == {"test-prompt-123": {"status": "success"}}
        mock_get.assert_called_once_with("http://localhost:8188/history/test-prompt-123", timeout=None)
    
    @patch('requests.get')
    def test_get_history_all_success(self, mock_get):
        """Test successful history retrieval for all prompts."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test-prompt-123": {"status": "success"}}
        mock_get.return_value = mock_response
        
        server = ComfyUIServer()
        result = server.get_history()
        
        assert result == {"test-prompt-123": {"status": "success"}}
        mock_get.assert_called_once_with("http://localhost:8188/history", timeout=None)


class TestWorkflow:
    """Test Workflow functionality with format detection."""
    
    def test_from_file_api_format(self, tmp_path):
        """Test loading API format workflow from file."""
        api_data = {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
        file_path = tmp_path / "workflow.json"
        
        with open(file_path, 'w') as f:
            json.dump(api_data, f)
        
        workflow = Workflow.from_file(str(file_path))
        
        assert workflow.api_json == api_data
        assert workflow.gui_json is None  # Should not create GUI structure
    
    def test_from_file_standard_format(self, tmp_path):
        """Test loading standard format workflow from file."""
        gui_data = {"nodes": [{"id": 6, "type": "CLIPTextEncode"}], "links": []}
        file_path = tmp_path / "workflow.json"
        
        with open(file_path, 'w') as f:
            json.dump(gui_data, f)
        
        workflow = Workflow.from_file(str(file_path))
        
        assert workflow.gui_json == gui_data
        assert workflow.api_json is None  # Should not create API structure
    
    def test_ensure_api_format_with_conversion(self, tmp_path):
        """Test ensuring API format when conversion is needed."""
        gui_data = {"nodes": [{"id": 6, "type": "CLIPTextEncode"}], "links": []}
        file_path = tmp_path / "workflow.json"
        
        with open(file_path, 'w') as f:
            json.dump(gui_data, f)
        
        workflow = Workflow.from_file(str(file_path))  # This will only populate gui_json
        
        with patch('comfy_commander.core.ComfyUIServer.is_available', return_value=True), \
             patch('comfy_commander.core.ComfyUIServer.convert_workflow') as mock_convert:
            mock_convert.return_value = {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
            
            server = ComfyUIServer()
            workflow.ensure_api_format(server)
            
            assert workflow.api_json == {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
            mock_convert.assert_called_once_with(gui_data)
    
    def test_ensure_api_format_server_unavailable(self, tmp_path):
        """Test ensuring API format when server is unavailable."""
        # Create a workflow that actually needs conversion (has nodes but no inputs)
        gui_data = {"nodes": [{"id": 6, "type": "CLIPTextEncode"}], "links": []}
        file_path = tmp_path / "workflow.json"
        
        with open(file_path, 'w') as f:
            json.dump(gui_data, f)
        
        workflow = Workflow.from_file(str(file_path))
        
        with patch('comfy_commander.core.ComfyUIServer.is_available', return_value=False):
            server = ComfyUIServer()
            with pytest.raises(ConnectionError, match="ComfyUI server is not available"):
                workflow.ensure_api_format(server)
    
    def test_execute_with_conversion(self, tmp_path):
        """Test workflow execution with automatic conversion."""
        gui_data = {"nodes": [{"id": 6, "type": "CLIPTextEncode"}], "links": []}
        file_path = tmp_path / "workflow.json"
        
        with open(file_path, 'w') as f:
            json.dump(gui_data, f)
        
        workflow = Workflow.from_file(str(file_path))
        
        with patch('comfy_commander.core.ComfyUIServer.is_available', return_value=True), \
             patch('comfy_commander.core.ComfyUIServer.convert_workflow') as mock_convert, \
             patch('comfy_commander.core.ComfyUIServer._send_workflow_to_server') as mock_execute:
            mock_convert.return_value = {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
            mock_execute.return_value = "test-prompt-123"
            
            server = ComfyUIServer()
            result = server.queue(workflow)
            
            assert result == "test-prompt-123"
            mock_convert.assert_called_once_with(gui_data)
            mock_execute.assert_called_once()
    
    def test_execute_without_conversion(self):
        """Test workflow execution when API format already exists."""
        api_data = {"6": {"inputs": {"text": "test"}, "class_type": "CLIPTextEncode"}}
        workflow = Workflow(api_json=api_data, gui_json={})
        
        with patch('comfy_commander.core.ComfyUIServer._send_workflow_to_server') as mock_execute:
            mock_execute.return_value = "test-prompt-123"
            
            server = ComfyUIServer()
            result = server.queue(workflow)
            
            assert result == "test-prompt-123"
            mock_execute.assert_called_once_with(api_data, "comfy-commander")