"""
Tests for ComfyOutput functionality including creation, saving, and node attributes.
"""

import pytest
import tempfile
import os
import json
import base64
from unittest.mock import Mock, patch
from PIL import Image
import io

from comfy_commander import Workflow, ComfyOutput, ComfyUIServer


class TestComfyOutput:
    """Test ComfyOutput creation and save functionality."""

    def test_comfy_output_creation_and_save(self):
        """Test ComfyOutput creation and save functionality."""
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create ComfyOutput
        comfy_output = ComfyOutput(
            data=img_data,
            filename="test.png",
            subfolder="output",
            type="output"
        )
        
        # Test that it's detected as an image
        assert comfy_output.is_image
        assert comfy_output.file_extension == "png"
        
        # Test saving to file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            comfy_output.save(tmp_path)
            
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

    def test_comfy_output_save_with_workflow_metadata(self):
        """Test ComfyOutput save functionality with workflow metadata embedding."""
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
            filename="test_with_workflow.png",
            subfolder="output",
            type="output"
        )
        comfy_output._workflow = test_workflow
        
        # Test saving to file with workflow metadata
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            comfy_output.save(tmp_path)
            
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

    def test_comfy_output_from_base64(self):
        """Test ComfyOutput creation from base64 data."""
        # Create test image data
        test_image = Image.new('RGB', (50, 50), color='blue')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Encode to base64
        base64_data = base64.b64encode(img_data).decode('utf-8')
        
        # Create ComfyOutput from base64
        comfy_output = ComfyOutput.from_base64(
            base64_data,
            filename="base64_test.png",
            subfolder="test",
            type="input"
        )
        
        # Verify properties
        assert comfy_output.filename == "base64_test.png"
        assert comfy_output.subfolder == "test"
        assert comfy_output.type == "input"
        assert len(comfy_output.data) > 0
        assert comfy_output.is_image

    def test_comfy_output_save_as(self):
        """Test ComfyOutput save_as method with automatic extension."""
        # Create test outputs with different file types
        png_output = ComfyOutput(data=b"fake_png_data", filename="test.png")
        mp4_output = ComfyOutput(data=b"fake_mp4_data", filename="test.mp4")
        wav_output = ComfyOutput(data=b"fake_wav_data", filename="test.wav")
        unknown_output = ComfyOutput(data=b"fake_data", filename="test.xyz")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test PNG output
            png_path = png_output.save_as(os.path.join(temp_dir, "my_image"))
            assert png_path.endswith(".png")
            assert os.path.exists(png_path)
            
            # Test MP4 output
            mp4_path = mp4_output.save_as(os.path.join(temp_dir, "my_video"))
            assert mp4_path.endswith(".mp4")
            assert os.path.exists(mp4_path)
            
            # Test WAV output
            wav_path = wav_output.save_as(os.path.join(temp_dir, "my_audio"))
            assert wav_path.endswith(".wav")
            assert os.path.exists(wav_path)
            
            # Test unknown output (should use original extension)
            unknown_path = unknown_output.save_as(os.path.join(temp_dir, "my_file"))
            assert unknown_path.endswith(".xyz")
            assert os.path.exists(unknown_path)

    def test_comfy_output_save_as_without_extension(self):
        """Test ComfyOutput save_as method when filename has no extension."""
        # Create outputs without file extensions but with proper data signatures
        # Create actual PNG data
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
        # Create actual WAV data
        wav_data = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        # Create actual MP4 data
        mp4_data = b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom'
        
        image_output = ComfyOutput(data=png_data, filename="test")
        video_output = ComfyOutput(data=mp4_data, filename="test")
        audio_output = ComfyOutput(data=wav_data, filename="test")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test image output (should default to .png)
            image_path = image_output.save_as(os.path.join(temp_dir, "my_image"))
            assert image_path.endswith(".png")
            assert os.path.exists(image_path)
            
            # Test video output (should default to .mp4)
            video_path = video_output.save_as(os.path.join(temp_dir, "my_video"))
            assert video_path.endswith(".mp4")
            assert os.path.exists(video_path)
            
            # Test audio output (should default to .wav)
            audio_path = audio_output.save_as(os.path.join(temp_dir, "my_audio"))
            assert audio_path.endswith(".wav")
            assert os.path.exists(audio_path)


class TestComfyOutputNodeAttribute:
    """Test ComfyOutput node attribute functionality."""
    
    def test_comfyoutput_creation_with_node(self, example_api_workflow_file_path):
        """Test creating ComfyOutput with node reference."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        node = workflow.node(id="31")  # KSampler node
        
        output_data = b"fake_output_data"
        output = ComfyOutput(
            data=output_data,
            filename="test.png",
            subfolder="output",
            type="output",
            node=node
        )
        
        assert output.node is not None
        assert output.node.id == "31"
        assert output.node.class_type == "KSampler"
        assert output.node.workflow == workflow
    
    def test_comfyoutput_creation_without_node(self):
        """Test creating ComfyOutput without node reference."""
        output_data = b"fake_output_data"
        output = ComfyOutput(
            data=output_data,
            filename="test.png",
            subfolder="output",
            type="output"
        )
        
        assert output.node is None
    
    def test_comfyoutput_from_base64_with_node(self, example_api_workflow_file_path):
        """Test creating ComfyOutput from base64 with node reference."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        node = workflow.node(id="31")  # KSampler node
        
        output_data = b"fake_output_data"
        base64_data = base64.b64encode(output_data).decode('utf-8')
        
        output = ComfyOutput.from_base64(
            base64_data=base64_data,
            filename="test.png",
            subfolder="output",
            type="output",
            node=node
        )
        
        assert output.node is not None
        assert output.node.id == "31"
        assert output.node.class_type == "KSampler"
    
    def test_comfyimage_from_base64_without_node(self):
        """Test creating ComfyOutput from base64 without node reference."""
        output_data = b"fake_output_data"
        base64_data = base64.b64encode(output_data).decode('utf-8')
        
        output = ComfyOutput.from_base64(
            base64_data=base64_data,
            filename="test.png",
            subfolder="output",
            type="output"
        )
        
        assert output.node is None
    
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
        output = images[0]
        
        # Check that node reference is set correctly
        assert output.node is not None
        assert output.node.id == "31"
        assert output.node.class_type == "KSampler"
        assert output.node.workflow == workflow
    
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
        output = images[0]
        
        # Check that node reference is None when no workflow provided
        assert output.node is None
    
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
        output = images[0]
        
        # Check that a basic Node object is created for non-existent node
        assert output.node is not None
        assert output.node.id == "999"
        assert output.node.workflow == workflow
        # The class_type should be empty since it's not in the workflow
        assert output.node.class_type == ""
    
    def test_comfyimage_node_access_properties(self, example_api_workflow_file_path):
        """Test accessing node properties through ComfyOutput."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        node = workflow.node(id="31")  # KSampler node
        
        output_data = b"fake_output_data"
        output = ComfyOutput(
            data=output_data,
            filename="test.png",
            subfolder="output",
            type="output",
            node=node
        )
        
        # Test accessing node properties through the output
        assert output.node.class_type == "KSampler"
        assert output.node.id == "31"
        
        # Test accessing node parameters
        seed_value = output.node.param("seed").value
        assert seed_value is not None
        
        # Test modifying node parameters through the output
        output.node.param("seed").set(999999)
        assert output.node.param("seed").value == 999999
        
        # Verify the change is reflected in the workflow
        assert workflow.node(id="31").param("seed").value == 999999
