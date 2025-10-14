"""
Tests for dual workflow synchronization between API and GUI JSON formats.
"""

import pytest

from comfy_commander import Workflow
from helpers import (
    assert_api_param_updated,
    assert_gui_widget_updated,
    assert_connections_preserved,
    assert_gui_connections_preserved
)


class TestWorkflowSynchronization:
    """Test dual workflow synchronization between API and GUI JSON."""

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
