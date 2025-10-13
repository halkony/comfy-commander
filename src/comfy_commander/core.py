"""
Core functionality for Comfy Commander.
"""

import json
from typing import Dict, Any, Optional

import attrs


@attrs.define
class PropertyAccessor:
    """Allows property access and assignment for node properties."""
    
    node: "Node" = attrs.field()
    property_name: str = attrs.field()
    
    def __eq__(self, other: Any) -> bool:
        """Compare property value for equality."""
        return self.node.inputs.get(self.property_name) == other
    
    def __ne__(self, other: Any) -> bool:
        """Compare property value for inequality."""
        return self.node.inputs.get(self.property_name) != other
    
    def __repr__(self) -> str:
        """String representation of the property value."""
        return repr(self.node.inputs.get(self.property_name))
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Handle assignment to the property accessor itself."""
        if name in ["node", "property_name"]:
            super().__setattr__(name, value)
        else:
            # This handles the case where we assign directly to the property accessor
            self.node.inputs[self.property_name] = value
    
    def __call__(self, value: Any = None) -> Any:
        """Allow direct assignment and value retrieval."""
        if value is not None:
            self.node.inputs[self.property_name] = value
            return value
        return self.node.inputs.get(self.property_name)
    
    def set(self, value: Any) -> None:
        """Set the property value."""
        self.node.inputs[self.property_name] = value
    
    @property
    def value(self) -> Any:
        """Get the property value."""
        return self.node.inputs.get(self.property_name)


@attrs.define
class Node:
    """Represents a single node in a ComfyUI workflow."""
    
    id: str = attrs.field()
    data: Dict[str, Any] = attrs.field()
    inputs: Dict[str, Any] = attrs.field(init=False)
    class_type: str = attrs.field(init=False)
    meta: Dict[str, Any] = attrs.field(init=False)
    
    def __attrs_post_init__(self):
        """Initialize derived fields after attrs initialization."""
        self.inputs = self.data.get("inputs", {})
        self.class_type = self.data.get("class_type", "")
        self.meta = self.data.get("_meta", {})
    
    def property(self, name: str) -> PropertyAccessor:
        """Get a property accessor for the node's inputs."""
        return PropertyAccessor(node=self, property_name=name)


@attrs.define
class Workflow:
    """Represents a ComfyUI workflow with nodes and their connections."""
    
    data: Dict[str, Any] = attrs.field()
    nodes: Dict[str, Node] = attrs.field(init=False)
    
    def __attrs_post_init__(self):
        """Initialize nodes after attrs initialization."""
        self.nodes = {}
        for node_id, node_data in self.data.items():
            self.nodes[node_id] = Node(id=node_id, data=node_data)
    
    @classmethod
    def from_file(cls, file_path: str) -> "Workflow":
        """Load a workflow from a JSON file."""
        with open(file_path, 'r') as f:
            workflow_data = json.load(f)
        return cls(data=workflow_data)
    
    def node(self, id: Optional[str] = None, name: Optional[str] = None) -> Node:
        """Get a node by ID or by class_type name."""
        if id is not None:
            if id in self.nodes:
                return self.nodes[id]
            raise KeyError(f"Node with ID '{id}' not found")
        
        if name is not None:
            for node in self.nodes.values():
                if node.class_type == name:
                    return node
            raise KeyError(f"Node with class_type '{name}' not found")
        
        raise ValueError("Either 'id' or 'name' must be provided")
