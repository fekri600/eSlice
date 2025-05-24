import csv
import json
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import itertools
import os
from SubstrateNetworkManager import SubstrateNetwork as SN
from VirtualNetworkManager import VirtualNodes as VN
from RequestManager import GenerateRequest as GReq

# Create necessary directories if they don't exist
os.makedirs('model_weights', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('json_files', exist_ok=True)
os.makedirs('csv_files', exist_ok=True)

# Define SubstrateNetworkEnvironment
class SubstrateNetworkEnvironment:
    def __init__(self, G, Gv, requests):
        self.G = G
        self.Gv = Gv
        self.requests = requests
        self.state = {node: data.get('hosted', []) for node, data in G.nodes(data=True)}

    def step(self, actions):
        next_state = self.state.copy()
        total_reward = 0
        done = False

        for action in actions:
            instance, action_type = action
            current_node = None
            for node, instances in self.state.items():
                if instance in instances:
                    current_node = node
                    break

            request = next((req for req in self.requests if req['instance'] == instance), None)
            if not request:
                continue

            if action_type == 0:  # Scale up action
                if self.can_scale_up(instance, current_node, request):
                    self.scale_up(instance, current_node, request)
                    total_reward += 1
                    request['status'] = 'scale_up'

            elif action_type == 1:  # Scale down action
                if self.can_scale_down(instance, current_node, request):
                    self.scale_down(instance, current_node, request)
                    total_reward += 1
                    request['status'] = 'scale_down'

            elif action_type == 2:  # Reallocate action
                non_endpoint_nodes = [node for node in self.G.nodes if not node.startswith('EP')]
                if not non_endpoint_nodes:
                    raise ValueError("No non-endpoint nodes available for selection")

                if 'PRB' in self.Gv.nodes[instance]:
                    target_nodes = [node for node in self.G.nodes if node.startswith('AP')]
                else:
                    target_nodes = non_endpoint_nodes

                if not target_nodes:
                    raise ValueError("No suitable nodes available for selection")

                target_node = random.choice(target_nodes)

                if self.can_reallocate(instance, current_node, target_node, request):
                    self.reallocate(instance, current_node, target_node, request)
                    total_reward += 1
                    request['status'] = 'reallocate'

        if all(not instances for instances in next_state.values()):
            done = True

        self.state = next_state
        return next_state, total_reward, done

    def can_scale_up(self, instance, node, request):
        required_cpu = request['CPU']
        required_ram = request['RAM']
        required_storage = request['Storage']
        required_bandwidth = request['Bandwidth']
        available_cpu = self.G.nodes[node]['CPU']
        available_ram = self.G.nodes[node]['RAM']
        available_storage = self.G.nodes[node]['Storage']
        available_bandwidth = request['available_bandwidth']
        if request['position'] == 0:
            required_prbs = request['PRBs']
            available_prbs = self.G.nodes[node]['PRB']
        else:
            available_prbs = required_prbs = 0
        return (available_cpu >= required_cpu and available_ram >= required_ram and
                available_storage >= required_storage and available_bandwidth >= required_bandwidth and
                available_prbs >= required_prbs)

    def scale_up(self, instance, node, request):
        self.G.nodes[node]['CPU'] -= request['CPU']
        self.G.nodes[node]['RAM'] -= request['RAM']
        self.G.nodes[node]['Storage'] -= request['Storage']
        if request['position'] == 0:
            self.G.nodes[node]['PRB'] -= request['PRBs']
        for i in range(len(request['substrate_path']) - 1):
            s_node1 = request['substrate_path'][i]
            s_node2 = request['substrate_path'][i + 1]
            if self.G.has_edge(s_node1, s_node2):
                self.G.edges[s_node1, s_node2]['Bandwidth'] -= request['Bandwidth']

    def can_scale_down(self, instance, node, request):
        return request['active'] == 0

    def scale_down(self, instance, node, request):
        self.G.nodes[node]['CPU'] += request['CPU']
        self.G.nodes[node]['RAM'] += request['RAM']
        self.G.nodes[node]['Storage'] += request['Storage']
        if request['position'] == 0:
            self.G.nodes[node]['PRB'] += request['PRBs']
        for i in range(len(request['substrate_path']) - 1):
            s_node1 = request['substrate_path'][i]
            s_node2 = request['substrate_path'][i + 1]
            if self.G.has_edge(s_node1, s_node2):
                self.G.edges[s_node1, s_node2]['Bandwidth'] += request['Bandwidth']

    def can_reallocate(self, instance, current_node, target_node, request):
        required_cpu = request['CPU']
        required_ram = request['RAM']
        required_storage = request['Storage']
        available_cpu = self.G.nodes[target_node]['CPU']
        available_ram = self.G.nodes[target_node]['RAM']
        available_storage = self.G.nodes[target_node]['Storage']
        if request['position'] == 0:
            required_prbs = request['PRBs']
            available_prbs = self.G.nodes[target_node]['PRB']
        else:
            available_prbs = required_prbs = 0
        return (available_cpu >= required_cpu and available_ram >= required_ram and
                available_storage >= required_storage and available_prbs >= required_prbs)

    def reallocate(self, instance, current_node, target_node, request):
        self.state[current_node].remove(instance)
        self.state[target_node].append(instance)
        self.G.nodes[target_node]['CPU'] -= request['CPU']
        self.G.nodes[target_node]['RAM'] -= request['RAM']
        self.G.nodes[target_node]['Storage'] -= request['Storage']
        self.G.nodes[current_node]['CPU'] += request['CPU']
        self.G.nodes[current_node]['RAM'] += request['RAM']
        self.G.nodes[current_node]['Storage'] += request['Storage']
        if request['position'] == 0:
            self.G.nodes[target_node]['PRB'] -= request['PRBs']
            self.G.nodes[current_node]['PRB'] += request['PRBs']

# Define DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define MultiAgentReSAA
class MultiAgentReSAA:
    def __init__(self, state_size, action_size, env, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.num_agents = num_agents
        self.agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
        self.episodes = 100
        self.batch_size = 32
        self.virtual_instances = list(itertools.chain(*self.env.state.values()))
        self.performance_log = []

        # Metrics initialization
        self.inter_slice_profit = 0
        self.inter_slice_ratio = 0
        self.reallocation_overhead = []
        self.reallocation_revenue = []
        self.agent_profits = [0] * num_agents
        self.penalty_resources = {'CPU': 0, 'RAM': 0, 'Storage': 0, 'Bandwidth': 0}
        self.allocation_attempts = 0
        self.successful_allocations = 0
        self.allocation_profits = []

    def train(self):
        for e in range(self.episodes):
            state = self.get_initial_state()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            episode_length = 0
            reallocations = 0
            successful_reallocations = 0

            for time in range(100):
                actions = []
                for idx, agent in enumerate(self.agents):
                    action_idx = agent.act(state)
                    action = self.decode_action(action_idx)
                    actions.append(action)

                next_state, reward, done = self.env.step(actions)
                total_reward += reward
                episode_length += 1

                reward = reward if not done else -10
                next_state_values = [len(next_state[node]) for node in self.env.G.nodes]
                next_state = np.reshape(next_state_values, [1, self.state_size])

                for idx, agent in enumerate(self.agents):
                    agent.remember(state, action_idx, reward, next_state, done)
                    self.agent_profits[idx] += reward

                reallocations += len([a for a in actions if a[1] == 2])
                successful_reallocations += sum(1 for a in actions if a[1] == 2 and reward > 0)

                self.allocation_attempts += len(actions)
                self.successful_allocations += sum(1 for a in actions if reward > 0)
                self.allocation_profits.append(reward)

                if reallocations > 0:
                    overhead = self.calculate_overhead(actions)
                    revenue = reward
                    self.reallocation_overhead.append(overhead)
                    self.reallocation_revenue.append(revenue)

                state = next_state
                if done:
                    print(
                        f"Episode: {e}/{self.episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}, Total Reward: {total_reward}")
                    break

                for agent in self.agents:
                    if len(agent.memory) > self.batch_size:
                        agent.replay(self.batch_size)

            self.performance_log.append({"episode": e, "total_reward": total_reward, "episode_length": episode_length})

            if e % 10 == 0:
                self.save_model_weights(e)

            self.inter_slice_profit += total_reward
            self.inter_slice_ratio = successful_reallocations / reallocations if reallocations > 0 else 0

            self.detect_penalties()

        return self.summarize_performance()

    def get_initial_state(self):
        state = [len(self.env.state[node]) for node in self.env.G.nodes]
        return state

    def decode_action(self, action_idx):
        instance_count = len(self.virtual_instances)
        action_type = action_idx // instance_count
        instance_idx = action_idx % instance_count
        instance = self.virtual_instances[instance_idx]
        return (instance, action_type)

    def save_model_weights(self, episode):
        for idx, agent in enumerate(self.agents):
            agent.model.save_weights(f"model_weights/agent_{idx}_episode_{episode}.h5")

    def calculate_overhead(self, actions):
        return sum(1 for action in actions if action[1] == 2)

    def detect_penalties(self):
        for node, data in self.env.G.nodes(data=True):
            if data.get('CPU', 0) < 0:
                self.penalty_resources['CPU'] += 1
            if data.get('RAM', 0) < 0:
                self.penalty_resources['RAM'] += 1
            if data.get('Storage', 0) < 0:
                self.penalty_resources['Storage'] += 1
            if data.get('Bandwidth', 0) < 0:
                self.penalty_resources['Bandwidth'] += 1

    def summarize_performance(self):
        def convert_numpy_int64(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            raise TypeError

        average_reallocation_overhead = np.mean(self.reallocation_overhead) if self.reallocation_overhead else 0
        average_reallocation_revenue = np.mean(self.reallocation_revenue) if self.reallocation_revenue else 0
        average_agent_profit = np.mean(self.agent_profits)
        allocation_ratio = self.successful_allocations / self.allocation_attempts if self.allocation_attempts > 0 else 0
        allocation_profit = np.sum(self.allocation_profits)

        performance_summary = {
            "total_episodes": self.episodes,
            "cumulative_reward": np.sum(self.allocation_profits),
            "average_reward_per_episode": np.mean(self.allocation_profits),
            "reward_variance": np.var(self.allocation_profits),
            "reward_standard_deviation": np.std(self.allocation_profits),
            "agent_rewards": self.agent_profits,
            "number_of_reallocated_nodes": self.successful_allocations,
            "number_of_nodes_that_cannot_be_reallocated": self.allocation_attempts - self.successful_allocations,
            "inter_slice_profit": self.inter_slice_profit,
            "inter_slice_ratio": self.inter_slice_ratio,
            "average_reallocation_overhead": average_reallocation_overhead,
            "average_reallocation_revenue": average_reallocation_revenue,
            "allocation_ratio": allocation_ratio,
            "allocation_profit": allocation_profit,
            "penalty_resources": self.penalty_resources,
            "sum_of_each_resource_endpoints": self.sum_resources(self.env.Gv),
            "sum_of_each_resource_substrate": self.sum_resources(self.env.G)
        }

        print("Performance Summary:", performance_summary)

        with open('json_files/performance_summary.json', 'w') as f:
            json.dump(performance_summary, f, default=convert_numpy_int64)
        with open('logs/performance_log.json', 'w') as f:
            json.dump(self.performance_log, f, default=convert_numpy_int64)

        return performance_summary

    def sum_resources(self, G):
        resources = {'CPU': 0, 'RAM': 0, 'Storage': 0, 'Bandwidth': 0, 'PRB': 0}
        for node, data in G.nodes(data=True):
            if node.startswith("EP"):
                continue
            for key in resources.keys():
                resources[key] += data.get(key, 0)
        return resources

    def evaluate(self, endpoint_range, step):
        results = []
        for num_endpoints in range(endpoint_range[0], endpoint_range[1] + step, step):
            substrate_network = SN(50, 0.1, 10, num_endpoints)
            G = substrate_network.G
            virtual_nodes = VN()
            Gv = virtual_nodes.Gv
            request_manager = GReq(G, Gv)
            requests = request_manager.create_requests_from_eps(G)
            env = SubstrateNetworkEnvironment(G, Gv, requests)

            total_requests = len(requests)
            scale_up_satisfied = 0
            reallocation_satisfied = 0
            rejected_requests = 0
            slice_utilization = {slice_id: 0 for slice_id in set(data['Slice_id'] for _, data in Gv.nodes(data=True))}

            for request in requests:
                if request['status'] == 'scale_up':
                    scale_up_satisfied += 1
                elif request['status'] == 'reallocate':
                    reallocation_satisfied += 1
                else:
                    rejected_requests += 1
                slice_id = Gv.nodes[request['instance']]['Slice_id']
                slice_utilization[slice_id] += (request['CPU'] + request['RAM'] + request['Storage'] + request['Bandwidth'])

            total_resources = sum(slice_utilization.values())
            for slice_id in slice_utilization:
                slice_utilization[slice_id] = (slice_utilization[slice_id] / total_resources) * 100

            slice_utilization['total'] = 100

            evaluation_result = {
                "number_of_endpoints": num_endpoints,
                "scale_up_satisfied": scale_up_satisfied,
                "reallocation_satisfied": reallocation_satisfied,
                "rejected_requests": rejected_requests,
                "slice_utilization": slice_utilization,
                "scale_up_benefits": scale_up_satisfied / total_requests,
                "reallocation_benefits": reallocation_satisfied / total_requests,
                "acceptance_ratio": (scale_up_satisfied + reallocation_satisfied) / total_requests
            }

            results.append(evaluation_result)

        with open('csv_files/evaluation_results.csv', 'w', newline='') as output_file:
            keys = results[0].keys()
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        return results

if __name__ == "__main__":
    num_endpoints = 100
    num_agents = 8

    substrate_network = SN(50, 0.1, 10, num_endpoints)
    G = substrate_network.G
    virtual_nodes = VN()
    Gv = virtual_nodes.Gv
    request_manager = GReq(G, Gv)
    requests = request_manager.create_requests_from_eps(G)
    env = SubstrateNetworkEnvironment(G, Gv, requests)

    state_size = len(G.nodes)
    action_size = len(list(itertools.chain(*env.state.values()))) * 3

    algorithm = MultiAgentReSAA(state_size, action_size, env, num_agents)
    algorithm.train()

    endpoint_range = (100, 1000)
    step = 100

    algorithm.evaluate(endpoint_range, step)
