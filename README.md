# 🤖 RemindMate: An Object-Finding Companion Agent with Followable Memory

Official repository containing the core memory mechanism algorithms and the real-world dataset for the paper **"RemindMate: An Object-Finding Companion Agent with Followable Memory"**.

> **Note:** This repository will provide the core logic and partial implementations of the RemindMate memory architecture (Memory Graph structure, Encoding, Storage, and Recall logic) to inspire further research. Due to privacy and hardware dependencies, the full deployment codebase (e.g., full robot navigation and hardware control) is not included for complete reproduction.

## 📖 Abstract

<p align="center">
  <img src="resources/Figure 1.png" alt="RemindMate Teaser" width="90%">
</p>

Finding an object, such as keys, wallets, or phones, we do not remember leaving somewhere occurs frequently. To address this issue, we present **RemindMate**, an object-finding companion agent that helps users recall where objects were placed by tracking human movement and capturing everyday placement events.

RemindMate centers on designing an agent brain that can efficiently encode visual events, support frequent memory updates, and enable rapid retrieval. We develop a brain architecture composed of:

* **Static Structure:** A hierarchical memory graph for organized memory management.

* **Dynamic Workflow:** A three-stage "Encoding-Storage-Recall" mechanism capturing object placement events, retaining structured memory traces, and retrieving relevant memory.

## 🧠 Architecture

### 1. RemindMate Workflow

<p align="center">
  <img src="resources/Figure 2.png" alt="RemindMate Workflow" width="90%">
</p>

RemindMate transforms continuous visual observations into a structured, queryable memory graph, and provides intuitive natural language reminders. It follows a dynamic brain workflow based on state transfer, implementing an Encoding-Storage-Recall mechanism.

### 2. System Implementation

<p align="center">
  <img src="resources/Figure 3.png" alt="System Implementation" width="90%">
</p>

The system integrates Long-term Memory (object/scene cognition, rules), Working Memory (continuous visual encoding and memory graph updating), and Natural Language Interaction. It processes continuous visual streams through multiple perception modules to update the structured memory graph.

## 🗂 Dataset & Experiments

<p align="center">
  <img src="resources/Figure 4.png" alt="Dataset Objects" width="80%">
</p>

To evaluate the performance and generalizability of RemindMate, we built a dedicated dataset from realistic indoor object placement and movement. The dataset features 10 volunteers interacting with 22 everyday objects in home-like environments.

The dataset is evaluated along four dimensions:

1. **Baseline**: One instance per type, coarse placement, single movement.

2. **Multi-Move**: One instance per type, coarse placement, multiple movements.

3. **Fine-Location**: One instance per type, fine-grained placement, single movement.

4. **Object-Variants**: Multiple instances per type, coarse placement, single movement.

<p align="center">
  <img src="resources/Figure 5.png" alt="Experimental Results" width="100%">
  <br>
  <em>Examples of our agent experiment results across the four dataset types.</em>
</p>

### 📥 Download the Dataset

> **⏳ Full Dataset Coming Soon!**
> We have compressed and uploaded a preview version of the dataset to the cloud drive. The complete dataset will be released shortly.

You can download the currently available data via the link below:

* **Quark Netdisk (夸克网盘):** [Link to Quark Netdisk](https://pan.quark.cn/s/1b1d5d288437) (Passcode: `fLWD`)

**Data Structure:**

```text
📦 RemindMate_Dataset
 ┣ 📂 Baseline
 ┣ 📂 Multi-Move
 ┣ 📂 Fine-Location
 ┗ 📂 Object-Variants
```

## 💻 Core Code

> **🚧 Core Code Coming Soon!**
> We are preparing to open-source a partial implementation focusing on the core logic of our memory mechanism (e.g., Memory Graph structure, Encoding, Storage, and Recall logic).
>
> **Note:** This will not be the full deployment codebase for complete reproduction (which involves specific hardware control and robot navigation), but rather key algorithmic snippets to inspire and assist further research.

### Prerequisites

* Python 3.8+

* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (for object detection)

* LLM/VLM APIs (e.g., Qwen-Max)

### Installation

To install the necessary dependencies, please run:

```bash
pip install -r requirements.txt
```

*(The core logic codebase will be published here shortly.)*

## 📝 Citation

If you find our work or dataset helpful, please consider citing our paper:

```bibtex
@article{remindmate202X,
  title={RemindMate: An Object-Finding Companion Agent with Followable Memory},
  author={Anonymous},
  journal={Conference/Journal Name},
  year={202X}
}
```

## 📧 Contact

For any questions regarding the dataset or the core logic, please open an issue or contact `[lann@stu.xidian.edu.cn]`.