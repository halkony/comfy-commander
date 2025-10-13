import pytest
from comfy_commander import Workflow
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
        workflow.api_json == snapshot
        workflow.gui_json == snapshot
    
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
