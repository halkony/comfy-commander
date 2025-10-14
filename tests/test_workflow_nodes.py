"""
Tests for Workflow node access and manipulation functionality.
"""

import pytest

from comfy_commander import Workflow


class TestWorkflowNodes:
    """Test Workflow node access and manipulation."""

    def test_workflow_node_editable_by_id(self, example_api_workflow_file_path):
        """Test accessing and editing nodes by ID."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        workflow.node(id="31").param("seed").set(1234567890)
        assert workflow.node(id="31").param("seed").value == 1234567890

    def test_workflow_node_editable_by_title(self, example_api_workflow_file_path):
        """Test accessing and editing nodes by title."""
        workflow = Workflow.from_file(example_api_workflow_file_path)
        workflow.node(title="KSampler").param("seed").set(1234567890)
        assert workflow.node(title="KSampler").param("seed").value == 1234567890

    def test_workflow_node_editable_by_class_type(self, example_api_workflow_file_path):
        """Test accessing and editing nodes by class type."""
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
