# eSlice
# ReSA-MARL: Resource Scalable Allocation via Multi-Agent Reinforcement Learning

This repository provides a modular simulation framework for dynamic resource allocation in 5G-enabled smart city infrastructures. It leverages Multi-Agent Reinforcement Learning (MARL) to allocate computing and network resources efficiently across heterogeneous service types like eMBB, URLLC, and mMTC.

---

## 📌 Overview

The simulation framework includes:
- A substrate network representing physical infrastructure.
- Virtual network hierarchical slices for samrt city applications.
- Smart city service request generation.
- Multi-agent Deep Q-Learning (DQN) based resource allocation.

It is designed for research and experimentation in intelligent network management, network slicing, and edge computing.

---

## 📁 Project Structure

| File                         | Description |
|------------------------------|-------------|
| `main.py`                    | Entry point that initializes and runs the simulation. |
| `ReSA-MARL.py`               | Core MARL environment and training loop. |
| `SubstrateNetworkManager.py` | Physical infrastructure and substrate topology. |
| `VirtualNetworkManager.py`   | Virtual slices, CNFs/SCFs, and hierarchical smart city services. |
| `RequestManager.py`          | Smart city service request modeling and traffic profiles. |
| `ResourcesManagement.py`     | Assumptions for CPU, RAM, storage, PRBs, and bandwidth needs. |
| `AllocationManager.py`       | (or Placement Manager) Resource mapping logic from virtual to substrate nodes. |

---

## ⚙️ Setup

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the full simulation using:

```bash
python main.py
```

This will:
- Generate a substrate network and virtual slices.
- Simulate smart city request traffic.
- Train DQN agents to perform multi-agent resource allocation.

---

## 🔧 Customize

This framework is modular and fully customizable:

- 🔹 Modify **hierarchical slices and virtual node setup** in  
  `VirtualNetworkManager.py`

- 🔹 Adjust the **substrate network's topology and resource capacities** in  
  `SubstrateNetworkManager.py`

- 🔹 Simulate different **service request types and traffic demands** in  
  `RequestManager.py`

- 🔹 Tune **QoS-based resource requirement formulas** (CPU, RAM, Storage, PRBs, Bandwidth) in  
  `ResourcesManagement.py`

This design allows you to simulate a wide range of 5G and smart city deployment scenarios.

---

## 📊 Outputs

- Trained models: `model_weights/`
- Logs and events: `logs/`
- Simulation results and resource mappings:  `json_files/`

---

## 🧪 Future Work

- Real-time visualization of virtual-to-substrate mappings.
- Graph theory based arguments for configurable topologies and resource allocation patterns.
- Pre-trained models for benchmarking.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

---

## 📬 Citation

comming soon.
