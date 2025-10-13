from comfy_commander import Workflow
import pytest
import os

@pytest.fixture
def api_workflow_file_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "flux_dev_checkpoint_example_api.json")

class TestWorkflows:
    def test_workflow_node_editable_by_id(self, snapshot, api_workflow_file_path):
        workflow = Workflow.from_file(api_workflow_file_path)
        workflow.node(id="31").property("seed").set(1234567890)
        assert workflow.node(id="31").property("seed").value == 1234567890
        assert workflow == snapshot

    def test_workflow_node_editable_by_name(self, snapshot, api_workflow_file_path):
        workflow = Workflow.from_file(api_workflow_file_path)
        workflow.node(name="KSampler").property("seed").set(1234567890)
        assert workflow.node(name="KSampler").property("seed").value == 1234567890
        assert workflow == snapshot
