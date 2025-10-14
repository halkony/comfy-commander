"""
Tests for MediaCollection functionality.
"""

import pytest
import tempfile
import os

from comfy_commander import Workflow, ComfyOutput, MediaCollection, ExecutionResult


class TestMediaCollection:
    """Test the MediaCollection class functionality."""
    
    def test_media_collection_iteration(self, example_api_workflow_file_path):
        """Test that MediaCollection can be iterated over like a list."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        # Create some test images with nodes
        node1 = workflow.node(id="31")  # KSampler node
        node2 = workflow.node(id="6")   # CLIPTextEncode node
        
        image1 = ComfyOutput(
            data=b"fake_image_data_1",
            filename="test1.png",
            node=node1
        )
        image2 = ComfyOutput(
            data=b"fake_image_data_2", 
            filename="test2.png",
            node=node2
        )
        
        # Create MediaCollection and add images
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
        
        image1 = ComfyOutput(
            data=b"fake_image_data_1",
            filename="test1.png",
            node=node1
        )
        image2 = ComfyOutput(
            data=b"fake_image_data_2",
            filename="test2.png", 
            node=node2
        )
        
        media = MediaCollection()
        media.append(image1)
        media.append(image2)
        
        # Test finding by title
        found_output = media.find_by_title("KSampler")
        assert found_output == image1
        assert found_output.node.title == "KSampler"
    
    def test_media_collection_find_by_title_no_match(self, example_api_workflow_file_path):
        """Test that find_by_title raises KeyError when no match is found."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        node = workflow.node(id="31")  # KSampler node
        output = ComfyOutput(
            data=b"fake_output_data",
            filename="test.png",
            node=node
        )
        
        media = MediaCollection()
        media.append(output)
        
        # Test that KeyError is raised for non-existent title
        with pytest.raises(KeyError, match="No output found with node title 'NonExistentTitle'"):
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
        
        image1 = ComfyOutput(
            data=b"fake_image_data_1",
            filename="test1.png",
            node=node1
        )
        image2 = ComfyOutput(
            data=b"fake_image_data_2",
            filename="test2.png",
            node=node2
        )
        
        media = MediaCollection()
        media.append(image1)
        media.append(image2)
        
        # Test that ValueError is raised for multiple matches
        with pytest.raises(ValueError, match="Multiple outputs found with node title 'Duplicate Title': 2 matches"):
            media.find_by_title("Duplicate Title")
    
    def test_media_collection_find_by_title_no_node(self):
        """Test that find_by_title handles images without nodes."""
        # Create image without a node
        image = ComfyOutput(
            data=b"fake_image_data",
            filename="test.png",
            node=None
        )
        
        media = MediaCollection()
        media.append(image)
        
        # Test that KeyError is raised when no node is present
        with pytest.raises(KeyError, match="No output found with node title 'SomeTitle'"):
            media.find_by_title("SomeTitle")
    
    def test_media_collection_extend(self):
        """Test extending MediaCollection with multiple images."""
        image1 = ComfyOutput(data=b"data1", filename="test1.png")
        image2 = ComfyOutput(data=b"data2", filename="test2.png")
        image3 = ComfyOutput(data=b"data3", filename="test3.png")
        
        media = MediaCollection()
        media.append(image1)
        media.extend([image2, image3])
        
        assert len(media) == 3
        assert media[0] == image1
        assert media[1] == image2
        assert media[2] == image3
    
    def test_media_collection_repr(self):
        """Test MediaCollection string representation."""
        media = MediaCollection()
        assert repr(media) == "MediaCollection(0 outputs)"
        
        image = ComfyOutput(data=b"data", filename="test.png")
        media.append(image)
        assert repr(media) == "MediaCollection(1 outputs)"
        
        media.append(image)
        assert repr(media) == "MediaCollection(2 outputs)"
    
    def test_execution_result_with_media_collection(self, example_api_workflow_file_path):
        """Test that ExecutionResult properly uses MediaCollection."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        
        node = workflow.node(id="31")
        output = ComfyOutput(
            data=b"fake_output_data",
            filename="test.png",
            node=node
        )
        
        media = MediaCollection()
        media.append(output)
        
        result = ExecutionResult(
            prompt_id="test_123",
            media=media,
            status="success"
        )
        
        # Test that we can iterate over result.media
        images_list = list(result.media)
        assert len(images_list) == 1
        assert images_list[0] == output
        
        # Test that we can find by title
        found_output = result.media.find_by_title("KSampler")
        assert found_output == output
        
        # Test that result.media is a MediaCollection
        assert isinstance(result.media, MediaCollection)
    
    def test_media_collection_filter_by_type(self):
        """Test filtering MediaCollection by output type."""
        # Create outputs with different types
        output1 = ComfyOutput(data=b"data1", filename="test1.png", type="output")
        output2 = ComfyOutput(data=b"data2", filename="test2.mp4", type="temp")
        output3 = ComfyOutput(data=b"data3", filename="test3.wav", type="output")
        output4 = ComfyOutput(data=b"data4", filename="test4.gif", type="temp")
        
        media = MediaCollection()
        media.extend([output1, output2, output3, output4])
        
        # Test filtering by type
        output_media = media.filter_by_type("output")
        temp_media = media.filter_by_type("temp")
        
        assert len(output_media) == 2
        assert len(temp_media) == 2
        
        # Test convenience properties
        assert media.output_media == output_media
        assert media.temp_media == temp_media
        
        # Verify the correct outputs are in each collection
        assert output1 in output_media
        assert output3 in output_media
        assert output2 in temp_media
        assert output4 in temp_media
