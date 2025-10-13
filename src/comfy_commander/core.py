"""
Core functionality for Comfy Commander.
"""

import json
import requests
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
    def from_file(cls, file_path: str, server: Optional["ComfyUIServer"] = None) -> "Workflow":
        """Load a workflow from a JSON file, automatically detecting format.
        
        Args:
            file_path: Path to the workflow JSON file
            server: Optional ComfyUI server instance for conversion if needed
            
        Returns:
            Workflow instance with both API and GUI data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Detect if this is a standard workflow (has 'nodes' and 'links' keys)
        if 'nodes' in data and 'links' in data:
            # Standard workflow format
            gui_data = data
            if server:
                # Convert to API format using server
                api_data = server.convert_workflow(gui_data)
            else:
                # Create minimal API structure (will need server for actual conversion)
                api_data = cls._create_minimal_api_from_gui(gui_data)
        else:
            # API workflow format
            api_data = data
            gui_data = cls._create_gui_from_api(api_data)
        
        return cls(api_json=api_data, gui_json=gui_data)
    
    @classmethod
    def _create_minimal_api_from_gui(cls, gui_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a minimal API structure from GUI data (placeholder until server conversion)."""
        api_data = {}
        for node in gui_data.get("nodes", []):
            node_id = str(node["id"])
            api_data[node_id] = {
                "class_type": node.get("type", ""),
                "inputs": {},
                "_meta": {"title": node.get("title", "")}
            }
        return api_data
    
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
    
    def _find_nodes_by_title(self, title: str) -> List[str]:
        """Find all node IDs that match the given title."""
        matching_node_ids = []
        for node_id, node_data in self.api_json.items():
            if node_data.get("_meta", {}).get("title") == title:
                matching_node_ids.append(node_id)
        return matching_node_ids
    
    def _find_nodes_by_class_type(self, class_type: str) -> List[str]:
        """Find all node IDs that match the given class_type."""
        matching_node_ids = []
        for node_id, node_data in self.api_json.items():
            if node_data.get("class_type") == class_type:
                matching_node_ids.append(node_id)
        return matching_node_ids
    
    def _find_nodes_by_name(self, name: str) -> List[str]:
        """Find all node IDs that match the given name (class_type)."""
        return self._find_nodes_by_class_type(name)
    
    def _create_node_from_id(self, node_id: str) -> Node:
        """Create a Node object from a node ID."""
        return Node(id=node_id, workflow=self)
    
    def _create_nodes_from_ids(self, node_ids: List[str]) -> List[Node]:
        """Create a list of Node objects from a list of node IDs."""
        return [self._create_node_from_id(node_id) for node_id in node_ids]

    def node(self, id: Optional[str] = None, name: Optional[str] = None, 
             title: Optional[str] = None, class_type: Optional[str] = None) -> Node:
        """Get a node by ID, name, title, or class_type."""
        if id is not None:
            if id in self.api_json:
                return self._create_node_from_id(id)
            raise KeyError(f"Node with ID '{id}' not found")
        
        if name is not None:
            matching_node_ids = self._find_nodes_by_name(name)
            if len(matching_node_ids) == 0:
                raise KeyError(f"Node with class_type '{name}' not found")
            elif len(matching_node_ids) > 1:
                raise ValueError(f"Multiple nodes found with class_type '{name}': {matching_node_ids}. Use node ID to specify which one.")
            else:
                return self._create_node_from_id(matching_node_ids[0])
        
        if title is not None:
            matching_node_ids = self._find_nodes_by_title(title)
            if len(matching_node_ids) == 0:
                raise KeyError(f"Node with title '{title}' not found")
            elif len(matching_node_ids) > 1:
                raise ValueError(f"Multiple nodes found with title '{title}': {matching_node_ids}. Use node ID to specify which one.")
            else:
                return self._create_node_from_id(matching_node_ids[0])
        
        if class_type is not None:
            matching_node_ids = self._find_nodes_by_class_type(class_type)
            if len(matching_node_ids) == 0:
                raise KeyError(f"Node with class_type '{class_type}' not found")
            elif len(matching_node_ids) > 1:
                raise ValueError(f"Multiple nodes found with class_type '{class_type}': {matching_node_ids}. Use node ID to specify which one.")
            else:
                return self._create_node_from_id(matching_node_ids[0])
        
        raise ValueError("One of 'id', 'name', 'title', or 'class_type' must be provided")
    
    def nodes(self, title: Optional[str] = None, class_type: Optional[str] = None) -> List[Node]:
        """Get all nodes that match the given title or class_type.
        
        Args:
            title: Title to match (exact match)
            class_type: Class type to match (exact match)
            
        Returns:
            List of Node objects that match the criteria
            
        Raises:
            ValueError: If neither title nor class_type is provided
        """
        if title is None and class_type is None:
            raise ValueError("Either 'title' or 'class_type' must be provided")
        
        matching_node_ids = set()
        
        if title is not None:
            matching_node_ids.update(self._find_nodes_by_title(title))
        
        if class_type is not None:
            matching_node_ids.update(self._find_nodes_by_class_type(class_type))
        
        return self._create_nodes_from_ids(list(matching_node_ids))
    
    def ensure_api_format(self, server: "ComfyUIServer") -> None:
        """Ensure the workflow has proper API format by converting from GUI if needed.
        
        Args:
            server: ComfyUI server instance for conversion
            
        Raises:
            ConnectionError: If server is not available
            requests.RequestException: If conversion fails
        """
        # Check if we have a minimal API structure (from standard workflow without server)
        if not self.api_json or not any(
            node_data.get("inputs") for node_data in self.api_json.values()
        ):
            # Need to convert from GUI format
            if not server.is_available():
                raise ConnectionError("ComfyUI server is not available for workflow conversion")
            
            # Convert using server
            api_data = server.convert_workflow(self.gui_json)
            self.api_json = api_data
    
    def execute(self, server: "ComfyUIServer", client_id: str = "comfy-commander") -> str:
        """Execute this workflow on the server.
        
        Args:
            server: ComfyUI server instance
            client_id: Client identifier for the execution
            
        Returns:
            Prompt ID for tracking the execution
            
        Raises:
            ConnectionError: If server is not available
            requests.RequestException: If execution fails
        """
        # Ensure we have proper API format
        self.ensure_api_format(server)
        
        # Execute the workflow
        return server.execute_workflow(self.api_json, client_id)
    
    def __eq__(self, other: Any) -> bool:
        """Compare workflows for equality."""
        if not isinstance(other, Workflow):
            return False
        return self.api_json == other.api_json and self.gui_json == other.gui_json


@attrs.define
class ComfyUIServer:
    """Handles communication with a local ComfyUI server."""
    
    base_url: str = attrs.field(default="http://localhost:8188")
    timeout: int = attrs.field(default=30)
    
    def __attrs_post_init__(self):
        """Initialize after attrs initialization."""
        # Ensure base_url doesn't end with trailing slash
        self.base_url = self.base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if the ComfyUI server is available."""
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def convert_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a standard workflow to API format using the /workflow/convert endpoint.
        
        Args:
            workflow_data: Standard workflow JSON data
            
        Returns:
            API format workflow data
            
        Raises:
            requests.RequestException: If the conversion request fails
        """
        response = requests.post(
            f"{self.base_url}/workflow/convert",
            json=workflow_data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def execute_workflow(self, api_workflow: Dict[str, Any], client_id: str = "comfy-commander") -> str:
        """Execute an API format workflow on the server.
        
        Args:
            api_workflow: API format workflow data
            client_id: Client identifier for the execution
            
        Returns:
            Prompt ID for tracking the execution
            
        Raises:
            requests.RequestException: If the execution request fails
        """
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": api_workflow, "client_id": client_id},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["prompt_id"]
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get the current queue status from the server.
        
        Returns:
            Queue status information
            
        Raises:
            requests.RequestException: If the request fails
        """
        response = requests.get(f"{self.base_url}/queue", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_history(self, prompt_id: Optional[str] = None) -> Dict[str, Any]:
        """Get execution history from the server.
        
        Args:
            prompt_id: Optional specific prompt ID to get history for
            
        Returns:
            History information
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}/history"
        if prompt_id:
            url += f"/{prompt_id}"
        
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

