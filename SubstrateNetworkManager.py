import networkx as nx  # Import NetworkX library for creating and managing graphs
import random  # Import random module for generating random numbers
import matplotlib.pyplot as plt  # Import Matplotlib's pyplot module for visualization
from VirtualNetworkManager import Slices  # Import Slices class from VirtualNetworkManager module


class SubstrateNetwork:
    # Initialize the SubstrateNetwork class with default parameters
    def __init__(self, num_nodes=50, edge_prob=0.1, access_node_count=10, endpoint_count=10):
        """
        Initialize the SubstrateNetwork with default values.

        :param num_nodes: Number of nodes in the network, default is 50.
        :param edge_prob: Probability of an edge between any pair of nodes, default is 0.1.
        :param access_node_count: Number of access nodes, default is 10.
        :param endpoint_count: Number of endpoint nodes, default is 10.
        """
        self.G = nx.erdos_renyi_graph(num_nodes, edge_prob)  # Create a random graph using the Erdős-Rényi model
        self.access_node_count = access_node_count  # Number of access nodes
        self.endpoint_count = endpoint_count  # Number of endpoint nodes
        self.slices = Slices().slices  # Initialize slices from the Slices class
        self.initialize_network()  # Call method to initialize the network
        self.create_endpoint_nodes()  # Call method to create endpoint nodes

    # Method to initialize the network with node attributes and connectivity
    def initialize_network(self):
        node_name_mapping = {}  # Dictionary to map original node names to new names
        new_node_attributes = {}  # Dictionary to hold the new node attributes

        # Iterate through nodes in the graph
        for node in self.G.nodes():
            node_name = f"CN{node}"  # Create a new name for computational nodes (CN)
            node_name_mapping[node] = node_name  # Map original name to new name
            new_node_attributes[node_name] = {  # Assign attributes to computational nodes
                'Name': node_name,
                'CPU': 16000,
                'RAM': 250000,
                'Storage': 10000
            }

        # Select a subset of nodes to act as access points (AP)
        access_nodes = random.sample(list(self.G.nodes()), self.access_node_count)
        for node in access_nodes:
            node_name = f"AP{node}"  # Create a new name for access points (AP)
            node_name_mapping[node] = node_name  # Map original name to new name
            new_node_attributes[node_name] = {  # Assign attributes to access points
                'Name': node_name,
                'CPU': 100,
                'RAM': 250000,
                'Storage': 10000,
                'PRB': random.randint(25, 275)
            }

        # Relabel nodes with the new names
        self.G = nx.relabel_nodes(self.G, node_name_mapping, copy=False)

        # Set the new node attributes
        nx.set_node_attributes(self.G, new_node_attributes)

        # Assign bandwidth to edges
        for u, v in self.G.edges():
            self.G.edges[u, v]['Bandwidth'] = 1000

    # Method to create and configure endpoint nodes (EP)
    def create_endpoint_nodes(self):
        # List of access nodes in the network
        access_nodes = [node for node in self.G.nodes if self.G.nodes[node]['Name'].startswith("AP")]

        # Create endpoint nodes and connect them to access nodes
        for _ in range(self.endpoint_count):
            # Generate a unique node ID for the endpoint
            node_id = max(int(node[2:]) for node in self.G.nodes if node.startswith("EP")) + 1 if any(
                node.startswith("EP") for node in self.G.nodes) else 1
            node_name = f"EP{node_id}"  # Create a new name for the endpoint
            self.G.add_node(node_name)  # Add endpoint node to the graph
            self.G.nodes[node_name]['Name'] = node_name  # Assign name attribute
            self.G.nodes[node_name]['Slice'] = random.randint(1, 8)  # Assign a random slice ID
            self.G.nodes[node_name]['Reliability'] = 0.99  # Assign reliability attribute

            # Assign a service type based on the slice
            Slice_id = self.G.nodes[node_name]['Slice']
            service_type = random.choice(list(self.slices[Slice_id]["children"].values()))
            self.G.nodes[node_name]['Service_Type'] = service_type

            # Assign traffic volume and latency based on the service type
            traffic_volume, latency = self.assign_traffic_volume(service_type)
            self.G.nodes[node_name]['Traffic Volume'] = traffic_volume
            self.G.nodes[node_name]['Latency'] = latency

            # Connect the endpoint node to a random access node
            access_node = random.choice(access_nodes)
            self.G.add_edge(node_name, access_node, Bandwidth=100)

        # Connect access nodes to computational nodes (CN)
        cn_nodes = [node for node in self.G.nodes if self.G.nodes[node]['Name'].startswith("CN")]
        for access_node in access_nodes:
            num_connections = random.randint(1, len(cn_nodes))  # Random number of connections
            connected_cn_nodes = random.sample(cn_nodes, num_connections)
            for cn_node in connected_cn_nodes:
                self.G.add_edge(access_node, cn_node, Bandwidth=1000)

    # Method to assign traffic volume and latency based on service type
    def assign_traffic_volume(self, Service_Type):
        if Service_Type == 'mMTC':
            traffic_volume = random.randint(1, 20)
            latency = random.randint(50, 500)
        elif Service_Type == 'URLLC':
            traffic_volume = random.randint(21, 100)
            latency = random.randint(10, 100)
        else:  # eMBB
            traffic_volume = random.randint(101, 1000)
            latency = random.randint(1, 10)
        return traffic_volume, latency

    # Method to plot the network
    def plot_network(self):
        plt.figure(figsize=(12, 8))  # Set the figure size
        nx.draw(self.G, with_labels=True, node_size=50)  # Draw the network graph
        plt.show()  # Display the plot

    # Method to print sample node information for debugging
    def print_sample_nodes(self):
        endpoint_node = None
        access_node = None
        cn_node = None
        # Iterate through nodes to find samples of each type
        for node, data in self.G.nodes(data=True):
            if 'EP' in data['Name'] and not endpoint_node:
                endpoint_node = (node, data)
            elif 'AP' in data['Name'] and not access_node:
                access_node = (node, data)
            elif 'CN' in data['Name'] and not cn_node:
                cn_node = (node, data)
            if endpoint_node and access_node and cn_node:
                break

        # Print sample node information
        print("Sample Endpoint Node:", endpoint_node)
        print("Sample Access Node:", access_node)
        print("Sample CN Node:", cn_node)

# Example usage
# sn = SubstrateNetwork(num_nodes=50, edge_prob=0.1, access_node_count=10, endpoint_count=10)
# sn.print_sample_nodes()
# sn.plot_network()
