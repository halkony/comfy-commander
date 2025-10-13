Comfy Commander is a package for programmatically running ComfyUI workloads either locally or remotely
 - Edit any node and its values from Python
 - Supports Local and RunPod ComfyUI instances
 - Convert standard ComfyUI workflows to API format
 - Execute workflows on local ComfyUI servers

## Quickstart

### Basic Workflow Usage
```python
from comfy_commander import Workflow

# Load an API format workflow
workflow = Workflow.from_file("./image_workflow.json")

# Modify node parameters
sampler_node = workflow.node(id="31")
sampler_node.param("seed").set(1234567890)
sampler_node.param("steps").set(8)

# Access node properties
print(f"Current seed: {sampler_node.param('seed').value}")
```

### Direct Execution
```python
import asyncio
from comfy_commander import ComfyUIServer, Workflow

server = ComfyUIServer("http://localhost:8188")
workflow = Workflow.from_file("./workflow.json")

result = server.execute(workflow)
for i, image in enumerate(result.media):
    image.save(f"./image_{i}.png")

```

### Queue for Later Processing
```python
from comfy_commander import ComfyUIServer, Workflow

server = ComfyUIServer("http://localhost:8188")
workflow = Workflow.from_file("./workflow.json")

# Queue workflow and get prompt ID immediately
prompt_id = server.queue(workflow)
print(f"Workflow queued with ID: {prompt_id}")

# Later, check status or get results
history = server.get_history(prompt_id)
```

## Requirements

### For Standard Workflow Conversion
To convert standard ComfyUI workflows to API format, you need:

1. **ComfyUI running locally** on `http://localhost:8188`
2. **Workflow Converter Extension** installed:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint
   ```
   Then restart ComfyUI.

## Testing

### Unit Tests (Default)
```bash
# Runs all unit tests by default (e2e tests are in separate files)
pytest tests/ -v
```

### End-to-End Tests
```bash
# Make sure ComfyUI is running with the converter extension
python run_e2e_tests.py
```

Or run e2e tests directly:
```bash
pytest tests/e2e_test_local_server.py -v
```

### All Tests (Including E2E)
```bash
# Run all tests including e2e (requires running ComfyUI instance)
pytest tests/ tests/e2e_test_local_server.py -v
```