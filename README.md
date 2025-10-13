Comfy Commander is a package for programmatically running ComfyUI workloads either locally or remotely
 - Edit any node and its values from Python
 - Supports Local and RunPod ComfyUI instances

## Quickstart
```python
import asyncio
from comfy_commander import LocalServer, Workflow, random_seed

local_server = LocalServer("http://localhost:8188/")
workflow = Workflow.from_file("./image_workflow.json")

# sets a new seed everytime the workflow is run
workflow.node(id=3).property("seed") = random_seed()
# sets the prompt
workflow.node(name="Positive Prompt").property("text") = "A beautiful woman with blonde hair"

async def main():
    result = await workflow.run()
    for i, image in enumerate(result.media):
        image.save(f"path/image_{i}.png")
    
if __name__ == "main":
    asyncio.run(main)
```