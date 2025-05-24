import random  # Import the random module to generate random numbers
from itertools import cycle  # Import cycle from itertools for cyclic iteration
import networkx as nx  # Import the NetworkX library for creating and managing graphs
import matplotlib.pyplot as plt  # Import Matplotlib's pyplot module for visualization

# Assuming SubstrateNetwork and VirtualNodes are defined in these modules
from SubstrateNetworkManager import SubstrateNetwork as sn
from VirtualNetworkManager import VirtualNodes as vn


class PeSA:
    def __init__(self, G, Gv):
        """
        Initializes the PeSA class with references to the substrate network (G) and virtual nodes graph (Gv).
        Prepares the substrate and virtual network graphs for resource management operations.

        :param G: The substrate network graph.
        :param Gv: The virtual instances CNFs graph.
        """
        self.Gv = Gv  # Reference to the virtual instances CNFs graph
        self.G = G  # Reference to the substrate network graph
        self.initialize_hosted_attribute()  # Initialize the 'hosted' attribute for substrate nodes
        self.PeSAA()  # Perform the PeSAA allocation

    def initialize_hosted_attribute(self):
        """
        Initializes the 'hosted' attribute for each substrate node.
        This attribute will hold a list of virtual nodes hosted on each substrate node.
        """
        for node, data in self.G.nodes(data=True):
            data['hosted'] = []  # Initialize the 'hosted' list for each node

    def PeSAA(self):
        """
        Allocates initial virtual nodes to substrate nodes using a round-robin approach.
        It attempts to find suitable substrate nodes for each virtual node based on resource demands
        and excludes endpoint nodes from being considered as potential hosts.
        """
        # Prepare a list of substrate nodes, excluding endpoints
        substrate_nodes = [(node, data) for node, data in self.G.nodes(data=True) if
                           not data.get('Name', '').startswith('EP')]

        # Create a cycle iterator over the substrate nodes
        substrate_cycle = cycle(substrate_nodes)

        for v_node, v_attrs in self.Gv.nodes(data=True):
            allocated = False
            # Try to allocate on each substrate node in sequence
            for _ in range(len(substrate_nodes)):
                s_node, s_attrs = next(substrate_cycle)  # Get the next substrate node from the cycle
                # Check if this substrate node can accommodate the virtual node
                if s_attrs['CPU'] >= v_attrs['CPU'] and s_attrs['RAM'] >= v_attrs['RAM'] and s_attrs['Storage'] >= \
                        v_attrs['Storage']:
                    # Deduct the resources from the substrate node
                    self.G.nodes[s_node]['CPU'] -= v_attrs['CPU']
                    self.G.nodes[s_node]['RAM'] -= v_attrs['RAM']
                    self.G.nodes[s_node]['Storage'] -= v_attrs['Storage']
                    if s_node.startswith('AP'):
                        # Add an attribute 'PRB' to the virtual node and set a value of 1 to it
                        self.Gv.nodes[v_node]['PRB'] = 1
                        # Deduct it from the 'PRB' of the substrate node
                        self.G.nodes[s_node]['PRB'] -= 1
                    # Mark this virtual node as allocated
                    self.Gv.nodes[v_node]['allocated_to'] = s_node
                    # Add the virtual node to the hosted list of the substrate node
                    self.G.nodes[s_node]['hosted'].append(v_node)
                    allocated = True
                    break  # Break from the loop once allocated
            if not allocated:
                print(f"Failed to allocate virtual node {v_node}")
