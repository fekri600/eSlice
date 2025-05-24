import random  # Import the random module to generate random numbers
from ResourcesManagement import \
    ResourceAssumptionAndCalculaiton as rac  # Import ResourceAssumptionAndCalculaiton as rac
from AllocationManager import PeSA  # Import PeSA from AllocationManager
from SubstrateNetworkManager import SubstrateNetwork as sn  # Import SubstrateNetwork as sn
from VirtualNetworkManager import VirtualNodes as vn  # Import VirtualNodes as vn
import pandas as pd # Import pandas library to save the results as a CSV file


class GenerateRequest:
    def __init__(self, G, Gv):
        """
        Initializes the GenerateRequest class by setting up the substrate and virtual networks,
        as well as other required resources. It also starts the process of calculating endpoint demands.

        :param G: The substrate network graph.
        :param Gv: The virtual instances CNFs graph.
        """
        self.G = G  # Reference to the substrate network graph
        self.set_of_ep = [node for node, data in self.G.nodes(data=True) if
                          data.get('Name', '').startswith('EP')]  # List of endpoint nodes
        self.Gv = Gv  # Reference to the virtual instances CNFs graph
        self.PeSA = PeSA(self.G, self.Gv)  # Initialize the PeSA class
        self.rac = rac()  # Initialize the ResourceAssumptionAndCalculaiton class
        self.ep_demand_calculation()  # Calculate the resource demands for each endpoint


    def ep_demand_calculation(self):
        """
        Calculates the resource demands for each endpoint (EP) and updates the substrate network accordingly.
        """
        for ep in self.set_of_ep:
            ep_data = self.G.nodes[ep]
            # Computational Resources calculation for CNF and SCF
            cpu_cnf, ram_cnf, storage_cnf = (
                self.rac.cpu_requirement_for_cnf_and_scf(ep_data['Latency'], 0, ep_data['Traffic Volume']),
                self.rac.ram_requirement_for_cnf_and_scf(ep_data['Latency'], 0, ep_data['Traffic Volume']),
                self.rac.storage_requirement_for_cnf_and_scf(ep_data['Latency'], 0)
            )

            cpu_scf, ram_scf, storage_scf = (
                self.rac.cpu_requirement_for_cnf_and_scf(ep_data['Latency'], 1, ep_data['Traffic Volume']),
                self.rac.ram_requirement_for_cnf_and_scf(ep_data['Latency'], 1, ep_data['Traffic Volume']),
                self.rac.storage_requirement_for_cnf_and_scf(ep_data['Latency'], 1)
            )

            # PRBs and Bandwidth Calculation
            prbs = self.rac.prb_num_calculate(ep_data['Latency'])
            bandwidth = self.rac.bandwidth_calculate(ep_data['Traffic Volume'], ep_data['Latency'])

            # Update node data with calculated resources
            self.G.nodes[ep].update({
                'CPU_CNF': cpu_cnf,
                'CPU_SCF': cpu_scf,
                'RAM_CNF': ram_cnf,
                'RAM_SCF': ram_scf,
                'Storage_CNF': storage_cnf,
                'Storage_SCF': storage_scf,
                'PRBs': prbs,
                'Bandwidth': bandwidth
            })

            # Path assignment
            path = self.construct_path(ep)
            self.G.nodes[ep]['Path'] = path

    def construct_path(self, ep):
        """
        Constructs a path for the given endpoint (EP) by selecting nodes from the virtual network.
        Adds edges in the substrate and virtual networks for the selected path.

        :param ep: The endpoint node.
        :return: The constructed path.
        """
        # Determine path length based on latency assumptions
        path_length = self.rac.assumption_parameters(self.G.nodes[ep]['Latency'])['path_length']

        # Select Access Point (AP) connected to the EP
        ap = list(self.G.neighbors(ep))

        # Select CNFs and SCFs from the virtual network
        CNFs_Child_Slice = [node for node, data in self.Gv.nodes(data=True) if
                            data.get('Slice_id', '') == self.G.nodes[ep]['Slice'] and node.startswith(
                                'CNF') and data.get('Service_Type', '') == self.G.nodes[ep]['Service_Type']]
        CNFs_Parent_Slice = [node for node, data in self.Gv.nodes(data=True) if
                             data.get('Slice_id', '') == self.G.nodes[ep]['Slice'] and node.startswith(
                                 'CNF') and data.get('Location', '') == 'Parent Slice']
        SCFs = [node for node, data in self.Gv.nodes(data=True) if
                data.get('Slice_id', '') == self.G.nodes[ep]['Slice'] and node.startswith('SCF') and data.get(
                    'Location', '') == 'Parent Slice']

        # Select CNFs for the path
        if path_length - 3 <= len(CNFs_Child_Slice):
            CNFs = random.sample(CNFs_Child_Slice, path_length - 3)
        else:
            CNFs = CNFs_Child_Slice + random.sample(CNFs_Parent_Slice, path_length - 3 - len(CNFs_Child_Slice))

        # Select a CNF hosted by the AP and construct the CNFs and SCFs path
        CNFs_hosted_by_ap = self.G.nodes[ap[0]]['hosted']
        CNF_ap = random.choice(CNFs_hosted_by_ap)
        CNFs_SCF = [CNF_ap] + CNFs + [random.choice(SCFs)]
        path = CNFs_SCF

        # Get the substrate nodes corresponding to the virtual nodes in the path
        sNodes_of_paths_vNodes = [self.Gv.nodes[vNode]['allocated_to'] for vNode in CNFs_SCF if
                                  'allocated_to' in self.Gv.nodes[vNode]]

        # Add edges in the substrate graph for sNodes_of_paths_vNodes
        for i in range(len(sNodes_of_paths_vNodes) - 1):
            self.G.add_edge(sNodes_of_paths_vNodes[i], sNodes_of_paths_vNodes[i + 1], Bandwidth=999)

        # Add edges in the virtual graph for CNFs_SCF
        for i in range(len(CNFs_SCF) - 1):
            self.Gv.add_edge(path[i], path[i + 1])
        return path

    def create_requests_from_eps(self, G):
        """
        Creates requests based on endpoint nodes and their paths.

        :param G: The substrate network graph.
        :return: A list of requests.
        """
        requests = []
        for node, data in G.nodes(data=True):
            if 'Path' in data:
                path = data['Path']
                if path:
                    available_bandwidth = self.available_substrate_bandwidth(path)
                    substrate_path = self.get_substrate_path(path)
                    for i, instance in enumerate(path):
                        request = {
                            'ep': node,
                            'instance': instance,
                            'Slice':data.get('Slice'),
                            'Bandwidth': data.get('Bandwidth', 0),
                            'available_bandwidth': available_bandwidth,
                            'position': i,
                            'active': 1,
                            'substrate_path': [node] + substrate_path
                        }

                        if i == 0:  # First instance in the path
                            request.update({
                                'CPU': data.get('CPU_CNF', 0),
                                'RAM': data.get('RAM_CNF', 0),
                                'Storage': data.get('Storage_CNF', 0),
                                'PRBs': data.get('PRBs', 0),
                            })
                        elif i == len(path) - 1:  # Last instance in the path
                            request.update({
                                'CPU': data.get('CPU_SCF', 0),
                                'RAM': data.get('RAM_SCF', 0),
                                'Storage': data.get('Storage_SCF', 0),
                            })
                        else:  # Intermediate instances in the path
                            request.update({
                                'CPU': data.get('CPU_CNF', 0),
                                'RAM': data.get('RAM_CNF', 0),
                                'Storage': data.get('Storage_CNF', 0),
                            })

                        requests.append(request)
        return requests

    def available_substrate_bandwidth(self, path):
        """
        Finds the minimum bandwidth of the links in the given path of virtual nodes.

        :param path: A list of virtual nodes representing the path.
        :return: The minimum bandwidth of the links in the path.
        """
        min_bandwidth = float('inf')

        for i in range(len(path) - 1):
            v_node1 = path[i]
            v_node2 = path[i + 1]

            s_node1 = self.Gv.nodes[v_node1]['allocated_to']
            s_node2 = self.Gv.nodes[v_node2]['allocated_to']

            if self.G.has_edge(s_node1, s_node2):
                bandwidth = self.G.edges[s_node1, s_node2].get('Bandwidth', float('inf'))
                if bandwidth < min_bandwidth:
                    min_bandwidth = bandwidth
            else:
                raise ValueError(f"No link between substrate nodes {s_node1} and {s_node2}")

        return min_bandwidth

    def get_substrate_path(self, virtual_path):
        """
        Gets the path of substrate nodes corresponding to the given virtual path.

        :param virtual_path: A list of virtual nodes representing the path.
        :return: The corresponding path of substrate nodes.
        """
        substrate_path = []

        for v_node in virtual_path:
            if 'allocated_to' in self.Gv.nodes[v_node]:
                s_node = self.Gv.nodes[v_node]['allocated_to']
                substrate_path.append(s_node)
            else:
                raise ValueError(f"Virtual node {v_node} does not have 'allocated_to' attribute")

        return substrate_path


def get_request_value_for_instance(requests, instance, attribute):
    """
    Retrieves the value of a specific attribute for a given instance from the requests list.

    :param requests: A list of requests.
    :param instance: The instance for which the value is needed.
    :param attribute: The attribute whose value is needed.
    :return: The value of the specified attribute.
    """
    for request in requests:
        if request['instance'] == instance:
            return request.get(attribute, None)
    return None


def calculate_total_remaining_resources(G):
    total_cpu = 0
    total_ram = 0
    total_storage = 0
    total_bandwidth = 0
    total_prbs = 0

    for node, data in G.nodes(data=True):
        if data.get('Type') == 'EP':
            continue

        total_cpu += data.get('CPU', 0)
        total_ram += data.get('RAM', 0)
        total_storage += data.get('Storage', 0)
        total_bandwidth += data.get('Bandwidth', 0)
        total_prbs += data.get('PRBs', 0)

    return {
        'CPU': total_cpu,
        'RAM': total_ram,
        'Storage': total_storage,
        'Bandwidth': total_bandwidth,
        'PRBs': total_prbs
    }

def calculate_average_resources(resources):
    number_of_resources = len(resources)
    total_sum = sum(resources.values())
    if number_of_resources == 0:
        return 0

    return total_sum / number_of_resources

def calculate_resources_per_slice(requests):
    slice_resources = {}

    for request in requests:
        slice_id = request.get('Slice')

        if slice_id not in slice_resources:
            slice_resources[slice_id] = {
                'CPU': 0,
                'RAM': 0,
                'Storage': 0,
                'Bandwidth': 0,
                'PRBs': 0
            }

        slice_resources[slice_id]['CPU'] += request.get('CPU', 0)
        slice_resources[slice_id]['RAM'] += request.get('RAM', 0)
        slice_resources[slice_id]['Storage'] += request.get('Storage', 0)
        slice_resources[slice_id]['Bandwidth'] += request.get('Bandwidth', 0)
        slice_resources[slice_id]['PRBs'] += request.get('PRBs', 0)

    return slice_resources

def save_results_to_csv(file_name, slice_resources, total_remaining_resources):
    data = {'Slice': [], 'CPU': [], 'RAM': [], 'Storage': [], 'Bandwidth': [], 'PRBs': [], 'Average': [], 'Sum of Resources': []}

    for slice_id, resources in slice_resources.items():
        data['Slice'].append(slice_id)
        data['CPU'].append(resources['CPU'])
        data['RAM'].append(resources['RAM'])
        data['Storage'].append(resources['Storage'])
        data['Bandwidth'].append(resources['Bandwidth'])
        data['PRBs'].append(resources['PRBs'])
        data['Average'].append(calculate_average_resources(resources))
        data['Sum of Resources'].append(sum(resources.values()))

    total_requested_resources = {
        'CPU': sum(data['CPU']),
        'RAM': sum(data['RAM']),
        'Storage': sum(data['Storage']),
        'Bandwidth': sum(data['Bandwidth']),
        'PRBs': sum(data['PRBs'])
    }

    data['Slice'].append('Total Requested')
    data['CPU'].append(total_requested_resources['CPU'])
    data['RAM'].append(total_requested_resources['RAM'])
    data['Storage'].append(total_requested_resources['Storage'])
    data['Bandwidth'].append(total_requested_resources['Bandwidth'])
    data['PRBs'].append(total_requested_resources['PRBs'])
    data['Average'].append(calculate_average_resources(total_requested_resources))
    data['Sum of Resources'].append(sum(total_requested_resources.values()))

    total_remaining = calculate_total_remaining_resources(G)

    data['Slice'].append('Total Remaining')
    data['CPU'].append(total_remaining['CPU'])
    data['RAM'].append(total_remaining['RAM'])
    data['Storage'].append(total_remaining['Storage'])
    data['Bandwidth'].append(total_remaining['Bandwidth'])
    data['PRBs'].append(total_remaining['PRBs'])
    data['Average'].append(calculate_average_resources(total_remaining))
    data['Sum of Resources'].append(sum(total_remaining.values()))

    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)







# Uncomment the following lines to create and test an instance of GenerateRequest
sn = sn(num_nodes=50, edge_prob=0.1, access_node_count=10, endpoint_count=1000)
G = sn.G

vn = vn()
Gv = vn.Gv
req = GenerateRequest(G, Gv)


# print('print of Graph G.nodes(data=True):',G.nodes(data=True))
# print('print of Graph Gv.nodes(data=True):',Gv.nodes(data=True))
# Create requests from EPs
requests = req.create_requests_from_eps(G)
endpints = [node for node in G.nodes if node.startswith('EP')]
print(endpints)
ep_embb = [node for node, attr in G.nodes(data=True) if node.startswith('EP') and attr.get('Service_Type') == 'eMBB']
ep_mmtc = [node for node, attr in G.nodes(data=True) if node.startswith('EP') and attr.get('Service_Type') == 'mMTC']
ep_urllc = [node for node, attr in G.nodes(data=True) if node.startswith('EP') and attr.get('Service_Type') == 'URLLC']

print(len(ep_embb), len(ep_mmtc), len(ep_urllc),len(ep_embb)+ len(ep_mmtc)+ len(ep_urllc))
#print('print  of requests', requests)
ep_counts = []
total_ep = 0
for s in range(8):
    ep_slice = [node for node, attr in G.nodes(data=True) if node.startswith('EP') and attr.get('Slice') == s+1]
    ep_slice_embb = [node for node, attr in G.nodes(data=True) if node.startswith('EP') and attr.get('Slice') == s + 1 and attr.get('Service_Type') == 'eMBB']
    ep_slice_mmtc = [node for node, attr in G.nodes(data=True) if node.startswith('EP') and attr.get('Slice') == s + 1 and attr.get('Service_Type') == 'mMTC']
    ep_slice_urllc = [node for node, attr in G.nodes(data=True) if node.startswith('EP') and attr.get('Slice') == s + 1 and attr.get('Service_Type') == 'URLLC']

    print(f'number of EP in slice {s+1} is:', len(ep_slice))
    print(f'number of EP in slice {s+1} -eMBB is:', len(ep_slice_embb))
    print(f'number of EP in slice {s + 1} -mMTC is:', len(ep_slice_mmtc))
    print(f'number of EP in slice {s + 1} -URRLLC is:', len(ep_slice_urllc))
    # Add the results to the list
    ep_counts.append({
        'Slice': f'Slice {s + 1}',
        'Total EPs': len(ep_slice),
        'eMBB EPs': len(ep_slice_embb),
        'mMTC EPs': len(ep_slice_mmtc),
        'URLLC EPs': len(ep_slice_urllc)
    })

    total_ep += len(ep_slice)
print(total_ep)
# Save the results to a CSV file
df_ep_counts = pd.DataFrame(ep_counts)
df_ep_counts.to_csv('ep_counts_per_slice.csv', index=False)

# Example usage

slice_resources = calculate_resources_per_slice(requests)
save_results_to_csv('resources.csv', slice_resources, calculate_total_remaining_resources(G))
