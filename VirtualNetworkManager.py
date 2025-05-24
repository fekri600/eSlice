import networkx as nx  # Import NetworkX library for creating and managing graphs

class Slices:
    """
    The Slices class defines various network slices and their respective service types.
    Each slice has a unique identifier, a name, and associated service types.
    """
    def __init__(self):
        self.slices = {
            1: {"name": "Rescue Drone Slice", "children": {"1.1": "URLLC", "1.2": "mMTC"}},
            2: {"name": "Holographic Communication Slice", "children": {"2.1": "URLLC", "2.2": "eMBB"}},
            3: {"name": "Voice over IP (VoIP) Slice", "children": {"3.1": "URLLC"}},
            4: {"name": "HD Video Calls Slice", "children": {"4.1": "eMBB"}},
            5: {"name": "Stationary Terrestrial Cameras Slice", "children": {"5.1": "URLLC", "5.2": "mMTC"}},
            6: {"name": "Aerial Video Surveillance Slice", "children": {"6.1": "URLLC", "6.2": "mMTC", "6.3": "eMBB"}},
            7: {"name": "Smart Homes Slice", "children": {"7.1": "eMBB", "7.2": "mMTC"}},
            8: {"name": "Industrial Buildings Slice", "children": {"8.1": "eMBB", "8.2": "URLLC", "8.3": "mMTC"}},
        }

class VirtualNodes:
    """
    The VirtualNodes class manages virtual nodes for each network slice.
    It generates virtual nodes with specific attributes and organizes them according to the slice hierarchy.
    """
    def __init__(self):
        self.Gv = nx.Graph()  # Initialize an empty graph to hold virtual nodes
        self.slices = Slices().slices  # Create an instance of Slices and access its slices attribute
        self.generate_virtual_nodes()  # Generate virtual nodes based on slice information

    def generate_virtual_nodes(self):
        """
        Generate virtual nodes for each slice and their respective service types.
        Each slice has a set of parent and child nodes with specific attributes.
        """
        node_id = 1  # Initialize node ID counter
        for Slice_id, slice_info in self.slices.items():
            # Create parent slice nodes
            for _ in range(10):  # Assume 10 CNF and SCF nodes per parent slice
                self.Gv.add_node(f"CNF{node_id}", Slice_id=Slice_id, Slice_Name=slice_info["name"], Location="Parent Slice", Type="CNF", CPU=1, RAM=1, Storage=1)
                node_id += 1
                self.Gv.add_node(f"SCF{node_id}", Slice_id=Slice_id, Slice_Name=slice_info["name"], Location="Parent Slice", Type="SCF", CPU=1, RAM=1, Storage=1)
                node_id += 1

            # Create child slice nodes
            for child_id, service_type in slice_info["children"].items():
                for _ in range(10):  # Assume 10 nodes per child slice
                    self.Gv.add_node(f"CNF{node_id}", Slice_id=Slice_id, Slice_Name=f"{slice_info['name']} - {service_type}", Location=f"Child Slice {child_id}", Type="CNF", Service_Type=service_type, CPU=1, RAM=1, Storage=1)
                    node_id += 1

    def print_virtual_nodes(self):
        """
        Print all virtual nodes and their attributes.
        """
        for node, attrs in self.Gv.nodes(data=True):
            print(f"{node}: {attrs}")

    def print_virtual_nodes_by_slice(self):
        """
        Print virtual nodes organized by slice.
        """
        for Slice_id, slice_info in self.slices.items():
            print(f"Slice {Slice_id}: {slice_info['name']}")
            nodes_in_slice = [node for node, data in self.Gv.nodes(data=True) if data['Slice_id'] == Slice_id]
            if not nodes_in_slice:
                print("  No virtual nodes.")
                continue
            for node in nodes_in_slice:
                node_attrs = self.Gv.nodes[node]
                print(f"  Node {node}: {node_attrs}")
            print()  # Blank line for readability

# Example usage
# Create an instance of VirtualNodes
# vn = VirtualNodes()
#
# # List nodes based on criteria
# list_nodes = [node for node, data in vn.Gv.nodes(data=True) if data.get('Slice_id', '') == 8 and node.startswith('CNF') and data.get('Service_Type', '') == 'eMBB']
# print(list_nodes)
# print(len(list_nodes))
#
# # Print at
