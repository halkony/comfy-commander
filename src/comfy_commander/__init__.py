"""
Comfy Commander - A package for programmatically running ComfyUI workloads.

This package provides tools to edit ComfyUI nodes and their values from Python,
supporting both local and remote ComfyUI instances.
"""

from .core import Workflow, ComfyUIServer, ComfyOutput, ExecutionResult, MediaCollection, Node

__version__ = "0.1.0"
__all__ = ["Workflow", "ComfyUIServer", "ComfyOutput", "ExecutionResult", "MediaCollection", "Node"]
