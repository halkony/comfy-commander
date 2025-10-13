"""
Core functionality for Comfy Commander.
"""

import json
from typing import Dict, Any, Optional, List, Union

import attrs


@attrs.define
class PropertyAccessor:
    """Allows property access and assignment for node properties."""
    
    node: "Node" = attrs.field()
    property_name: str = attrs.field()
    
    def __eq__(self, other: Any) -> bool:
        """Compare property value for equality."""
        return self.node.get_property_value(self.property_name) == other
    
    def __ne__(self, other: Any) -> bool:
        """Compare property value for inequality."""
        return self.node.get_property_value(self.property_name) != other
    
    def __repr__(self) -> str:
        """String representation of the property value."""
        return repr(self.node.get_property_value(self.property_name))
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Handle assignment to the property accessor itself."""
        if name in ["node", "property_name"]:
            super().__setattr__(name, value)
        else:
            # This handles the case where we assign directly to the property accessor
            self.node.set_property_value(self.property_name, value)
    
    def __call__(self, value: Any = None) -> Any:
        """Allow direct assignment and value retrieval."""
        if value is not None:
            self.node.set_property_value(self.property_name, value)
            return value
        return self.node.get_property_value(self.property_name)
    
    def set(self, value: Any) -> None:
        """Set the property value."""
        self.node.set_property_value(self.property_name, value)
    
    @property
    def value(self) -> Any:
        """Get the property value."""
        return self.node.get_property_value(self.property_name)


@attrs.define
class Node:
    """Represents a single node in a ComfyUI workflow."""
    
    id: str = attrs.field()
    workflow: "Workflow" = attrs.field()
    
    def get_property_value(self, property_name: str) -> Any:
        """Get a property value from the API JSON format."""
        if self.id in self.workflow.api_json:
            return self.workflow.api_json[self.id].get("inputs", {}).get(property_name)
        return None
    
    def set_property_value(self, property_name: str, value: Any) -> None:
        """Set a property value in both API JSON and GUI JSON formats."""
        # Update API JSON
        if self.id in self.workflow.api_json:
            if "inputs" not in self.workflow.api_json[self.id]:
                self.workflow.api_json[self.id]["inputs"] = {}
            self.workflow.api_json[self.id]["inputs"][property_name] = value
        
        # Update GUI JSON - find the corresponding node and update widgets_values
        self.workflow._sync_property_to_gui(self.id, property_name, value)
    
    def param(self, name: str) -> PropertyAccessor:
        """Get a parameter accessor for the node's inputs."""
        return PropertyAccessor(node=self, property_name=name)
    
    @property
    def class_type(self) -> str:
        """Get the class type from API JSON."""
        if self.id in self.workflow.api_json:
            return self.workflow.api_json[self.id].get("class_type", "")
        return ""
    
    @property
    def title(self) -> str:
        """Get the title from API JSON metadata."""
        if self.id in self.workflow.api_json:
            return self.workflow.api_json[self.id].get("_meta", {}).get("title", "")
        return ""


@attrs.define
class Workflow:
    """Represents a ComfyUI workflow with nodes and their connections."""
    
    api_json: Dict[str, Any] = attrs.field()
    gui_json: Dict[str, Any] = attrs.field()
    
    def __attrs_post_init__(self):
        """Initialize after attrs initialization."""
        pass
    
    @classmethod
    def from_file(cls, file_path: str) -> "Workflow":
        """Load a workflow from a JSON file (API format)."""
        with open(file_path, 'r') as f:
            api_data = json.load(f)
        
        # Create a minimal GUI JSON structure
        gui_data = cls._create_gui_from_api(api_data)
        
        return cls(api_json=api_data, gui_json=gui_data)
    
    @classmethod
    def from_image(cls, file_path: str) -> "Workflow":
        """Load a workflow from an image metadata file."""
        with open(file_path, 'r') as f:
            image_data = json.load(f)
        
        # Extract prompt and workflow from image metadata
        api_data = image_data.get("prompt", {})
        gui_data = image_data.get("workflow", {})
        
        return cls(api_json=api_data, gui_json=gui_data)
    
    @classmethod
    def _create_gui_from_api(cls, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a minimal GUI JSON structure from API data."""
        nodes = []
        for node_id, node_data in api_data.items():
            # Convert API inputs to widgets_values
            widgets_values = []
            inputs = node_data.get("inputs", {})
            
            # Create a mapping of input names to their order
            input_order = list(inputs.keys())
            
            for input_name in input_order:
                value = inputs[input_name]
                # Skip connection inputs (lists with node references)
                if not isinstance(value, list) or len(value) != 2:
                    widgets_values.append(value)
            
            gui_node = {
                "id": int(node_id),
                "type": node_data.get("class_type", ""),
                "pos": [0, 0],  # Default position
                "size": [200, 100],  # Default size
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "title": node_data.get("_meta", {}).get("title", ""),
                "properties": {
                    "cnr_id": "comfy-core",
                    "ver": "0.3.64",
                    "Node name for S&R": node_data.get("class_type", ""),
                    "widget_ue_connectable": {}
                },
                "widgets_values": widgets_values
            }
            nodes.append(gui_node)
        
        return {
            "id": "generated-workflow",
            "revision": 0,
            "last_node_id": max([int(nid) for nid in api_data.keys()]) if api_data else 0,
            "last_link_id": 0,
            "nodes": nodes,
            "links": [],
            "groups": [],
            "config": {},
            "extra": {
                "ds": {"scale": 1.0, "offset": [0, 0]},
                "ue_links": [],
                "links_added_by_ue": [],
                "frontendVersion": "1.27.10"
            },
            "version": 0.4
        }
    
    def _sync_property_to_gui(self, node_id: str, property_name: str, value: Any) -> None:
        """Sync a property change from API JSON to GUI JSON."""
        # Find the corresponding node in GUI JSON
        for node in self.gui_json.get("nodes", []):
            if str(node["id"]) == node_id:
                # Find the position of this property in the widgets_values
                api_node = self.api_json.get(node_id, {})
                inputs = api_node.get("inputs", {})
                
                # Get the order of non-connection inputs
                input_order = []
                for input_name, input_value in inputs.items():
                    # Skip connection inputs (lists with node references)
                    if not isinstance(input_value, list) or len(input_value) != 2:
                        input_order.append(input_name)
                
                # Find the index of this property
                try:
                    property_index = input_order.index(property_name)
                    
                    # For KSampler, we need to account for the "randomize" value at index 1
                    if api_node.get("class_type") == "KSampler" and property_index > 0:
                        property_index += 1
                    
                    # Ensure widgets_values is long enough
                    while len(node["widgets_values"]) <= property_index:
                        node["widgets_values"].append(None)
                    
                    # Update the value at the correct position
                    node["widgets_values"][property_index] = value
                except ValueError:
                    # Property not found in inputs, skip
                    pass
                break
    
    def node(self, id: Optional[str] = None, name: Optional[str] = None, 
             title: Optional[str] = None, class_type: Optional[str] = None) -> Node:
        """Get a node by ID, name, title, or class_type."""
        if id is not None:
            if id in self.api_json:
                return Node(id=id, workflow=self)
            raise KeyError(f"Node with ID '{id}' not found")
        
        if name is not None:
            for node_id, node_data in self.api_json.items():
                if node_data.get("class_type") == name:
                    return Node(id=node_id, workflow=self)
            raise KeyError(f"Node with class_type '{name}' not found")
        
        if title is not None:
            for node_id, node_data in self.api_json.items():
                if node_data.get("_meta", {}).get("title") == title:
                    return Node(id=node_id, workflow=self)
            raise KeyError(f"Node with title '{title}' not found")
        
        if class_type is not None:
            matching_nodes = []
            for node_id, node_data in self.api_json.items():
                if node_data.get("class_type") == class_type:
                    matching_nodes.append(node_id)
            
            if len(matching_nodes) == 0:
                raise KeyError(f"Node with class_type '{class_type}' not found")
            elif len(matching_nodes) > 1:
                raise ValueError(f"Multiple nodes found with class_type '{class_type}': {matching_nodes}. Use node ID to specify which one.")
            else:
                return Node(id=matching_nodes[0], workflow=self)
        
        raise ValueError("One of 'id', 'name', 'title', or 'class_type' must be provided")
    
    def __eq__(self, other: Any) -> bool:
        """Compare workflows for equality."""
        if not isinstance(other, Workflow):
            return False
        return self.api_json == other.api_json and self.gui_json == other.gui_json