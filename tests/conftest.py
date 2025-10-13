"""
Pytest configuration and fixtures for Comfy Commander tests.
"""

import pytest
from pathlib import Path


@pytest.fixture
def example_api_workflow_file_path():
    return Path(__file__).parent / "fixtures" / "flux_dev_checkpoint_example_api.json"


@pytest.fixture
def example_standard_workflow_file_path():
    return Path(__file__).parent / "fixtures" / "flux_dev_checkpoint_example_standard.json"


@pytest.fixture
def example_image_file_path():
    return Path(__file__).parent / "fixtures" / "image_workflow.json"
