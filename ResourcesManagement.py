import random  # Import the random module to generate random numbers
import numpy as np  # Import the numpy library for numerical operations

class ResourceAssumptionAndCalculaiton:
    """
    Class to assume and calculate various resource parameters based on latency and service type.
    """

    def __init__(self):
        pass  # The constructor does not perform any initialization

    def assumption_parameters(self, latency):
        """
        Determine the service type and generate various attributes based on latency.

        :param latency: The latency value to determine the service type and generate attributes.
        :return: A dictionary containing the generated attributes.
        """
        service_types = ['URLLC', 'mMTC', 'eMBB']  # Define the service types
        # Determine the service type based on latency
        if latency <= 1:
            service_type = service_types[0]
        elif 2 <= latency <= 10:
            service_type = service_types[2]
        else:
            service_type = service_types[1]

        # Determine the packet size based on service type
        packet_size = random.randint(256, 2048) if service_type == 'URLLC' else random.randint(160, 1024) if service_type == 'mMTC' else random.randint(8192, 32768)

        # Additional attributes based on service type
        attributes = {
            'service_type': service_type,
            'path_length': random.randint(7, 15),
            'processing_time': round(random.uniform(0.01, 0.05), 4),
            'static_overhead_memory': random.randint(512, 2048),
            'additional_memory_scf': random.randint(1024, 4096),
            'scaling_factor': random.uniform(1.2, 1.5) if service_type == 'URLLC' else random.uniform(0.5, 1) if service_type == 'mMTC' else random.uniform(1.6, 2),
            'memory_usage': random.uniform(0.1, 1) if service_type == 'URLLC' else random.uniform(0.001, 0.09) if service_type == 'mMTC' else random.uniform(2, 10),
            'binary_variable_kappa': random.choice([0, 1]),
            'baseline_storage_requirements_CNF': random.randint(1, 10),
            'baseline_storage_requirements_SCF': random.randint(11, 50),
            'datasets': random.randint(5, 10),
            'traffic_arrival_rate': random.randint(1, 10) if service_type == 'URLLC' else random.randint(10, 50) if service_type == 'mMTC' else random.randint(5, 20),
            'node_reliability_score': round(random.uniform(0.9, 1.0), 2),
            'link_reliability_score': round(random.uniform(0.95, 0.99), 2),
            'packet_size': packet_size,
            'component_carriers': random.randint(1, 16),
            'max_code_rate': random.choice(['1/3', '1/2', '2/3', '3/4', '4/5']),
            'mimo_layers': 8,
            'modulation_order': random.choice([2, 4, 6]),
            'numerology': random.choice([0, 1, 2, 3, 4, 5]),
            'overhead': round(random.uniform(0.1, 0.3), 2),
            'PRB_per_node': random.randint(25, 275),
            'traffic_volume': random.randint(1, 20) if service_type == 'mMTC' else random.randint(10, 100) if service_type == 'URLLC' else random.randint(1000, 5000)
        }

        return attributes

    def link_delay(self, packet_size, latency, path):
        """
        Calculate the link round-trip time (RTT) based on packet size, latency, and path length.

        :param packet_size: Size of the packet.
        :param latency: The latency value.
        :param path: The path length.
        :return: The calculated link RTT.
        """
        # Calculate the link RTT based on the given formula
        link_rtt = abs((latency / (len(path) + 1) / 2) - (1 * self.assumption_parameters(latency)['processing_time'] +
                      (self.assumption_parameters(latency)['processing_time'] /
                       (1 - (self.assumption_parameters(latency)['traffic_arrival_rate'] * self.assumption_parameters(latency)['processing_time'])))))
        return link_rtt

    def cpu_requirement_for_cnf_and_scf(self, latency, kappa, traffic_volume):
        """
        Calculate the CPU requirement for CNF and SCF based on latency, kappa, and traffic volume.

        :param latency: The latency value.
        :param kappa: The kappa value (0 for net, 1 for app).
        :param traffic_volume: The traffic volume.
        :return: The calculated CPU requirement.
        """
        params = self.assumption_parameters(latency)
        phi_cpu = 0.5
        phi_hat_cpu = 0.1
        return ((1 / params['processing_time'] + phi_cpu * traffic_volume + phi_hat_cpu * kappa) / 100)

    def ram_requirement_for_cnf_and_scf(self, latency, kappa, traffic_volume):
        """
        Calculate the RAM requirement for CNF and SCF based on latency, kappa, and traffic volume.

        :param latency: The latency value.
        :param kappa: The kappa value (0 for net, 1 for app).
        :param traffic_volume: The traffic volume.
        :return: The calculated RAM requirement.
        """
        params = self.assumption_parameters(latency)
        return (params['static_overhead_memory'] + params['additional_memory_scf'] * kappa +
                params['scaling_factor'] * params['memory_usage'] * traffic_volume * params['processing_time'])

    def storage_requirement_for_cnf_and_scf(self, latency, kappa):
        """
        Calculate the storage requirement for CNF and SCF based on latency and kappa.

        :param latency: The latency value.
        :param kappa: The kappa value (0 for net, 1 for app).
        :return: The calculated storage requirement.
        """
        params = self.assumption_parameters(latency)
        storage = params['baseline_storage_requirements_CNF'] + params['baseline_storage_requirements_SCF'] * kappa
        for _ in range(params['datasets']):
            storage += 0.5  # Assuming each dataset adds 50MB to the storage requirement
        return storage

    def calculate_theta_prb(self, latency):
        """
        Calculate the theta PRB (Physical Resource Block) based on latency.

        :param latency: The latency value.
        :return: The calculated theta PRB.
        """
        params = self.assumption_parameters(latency)
        J = params['component_carriers']
        R_max = 3 / 4
        v_Layer = [params['mimo_layers']] * J
        Q_m = [params['modulation_order']] * J
        f = [params['scaling_factor']] * J
        μ = params['numerology']
        OH = [params['overhead']] * J
        T_s_mu = 1e-3 / (2 ** μ)
        theta_prb = (12 * R_max * 10 ** (-6)) / T_s_mu * sum(v_Layer[j] * Q_m[j] * f[j] * (1 - OH[j]) for j in range(J))

        return theta_prb

    def prb_num_calculate(self, latency):
        """
        Calculate the number of PRBs (Physical Resource Blocks) required based on latency.

        :param latency: The latency value.
        :return: The calculated number of PRBs.
        """
        service_type = self.assumption_parameters(latency)['service_type']
        packet_size = self.assumption_parameters(latency)['packet_size']
        theta_prb = self.calculate_theta_prb(latency)
        min_theta_ch = 100000  # Placeholder for minimum data rate required by eMBB services
        L_s_ch = 1  # Placeholder for maximum latency threshold for URLLC services
        if service_type == 'eMBB':
            return np.ceil(min_theta_ch / theta_prb)
        elif service_type == 'URLLC':
            return np.ceil(packet_size / (L_s_ch * theta_prb))
        else:  # mMTC
            return 1  # A single PRB is often adequate for mMTC applications

    def bandwidth_calculate(self, traffic_volume, latency):
        """
        Calculate the bandwidth required based on traffic volume and latency.

        :param traffic_volume: The traffic volume.
        :param latency: The latency value.
        :return: The calculated bandwidth in Mbps.
        """
        params = self.assumption_parameters(latency)
        packet_size = params['packet_size']
        bandwidth = (traffic_volume * packet_size) / (latency / 2)  # Bandwidth in bits per second
        bandwidth = bandwidth / 1e6  # Convert bps to Mbps
        return bandwidth
