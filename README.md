# **🤖 RemindMate: An Object-Finding Companion Agent with Followable Memory**

Official repository containing the core memory mechanism algorithms and the real-world dataset for the paper **"RemindMate: An Object-Finding Companion Agent with Followable Memory"**.

**Note:** This repository provides the core logic and partial implementations of the RemindMate memory architecture (Memory Graph structure, Encoding, Storage, and Recall logic) to inspire further research. Due to privacy and hardware dependencies, the full deployment codebase (e.g., full robot navigation and hardware control) is not included for complete reproduction.

## **📖 Abstract**

Finding an object, such as keys, wallets, or phones, we do not remember leaving somewhere occurs frequently. To address this issue, we present **RemindMate**, an object-finding companion agent that helps users recall where objects were placed by tracking human movement and capturing everyday placement events.

RemindMate centers on designing an agent brain that can efficiently encode visual events, support frequent memory updates, and enable rapid retrieval. We develop a brain architecture composed of:

* **Static Structure:** A hierarchical memory graph for organized memory management.  
* **Dynamic Workflow:** A three-stage "Encoding-Storage-Recall" mechanism capturing object placement events, retaining structured memory traces, and retrieving relevant memory.

## **🧠 Architecture**

### **1\. RemindMate Workflow**

RemindMate transforms continuous visual observations into a structured, queryable memory graph, and provides intuitive natural language reminders.

### **2\. System Implementation**

The system integrates Long-term Memory (object/scene cognition, rules), Working Memory (continuous visual encoding and memory graph updating), and Natural Language Interaction.

## **🗂 Dataset**

To evaluate the performance and generalizability of RemindMate, we built a dedicated dataset from realistic indoor object placement and movement. The dataset features 10 volunteers interacting with 22 everyday objects in home-like environments.

The dataset is evaluated along four dimensions:

1. **Baseline**: One instance per type, coarse placement, single movement.  
2. **Multi-Move**: One instance per type, coarse placement, multiple movements.  
3. **Fine-Location**: One instance per type, fine-grained placement, single movement.  
4. **Object-Variants**: Multiple instances per type, coarse placement, single movement.

### **📥 Download the Dataset**

We have compressed the dataset and uploaded it to a cloud drive. You can download it via the link below:

* **Google Drive:** [\[Link to Google Drive\]](#bookmark=id.uwxsjuwgdc6j)  
* **Baidu Netdisk (百度网盘):** [\[Link to Baidu Pan\]](#bookmark=id.uwxsjuwgdc6j) (Passcode: xxxx)

**Data Structure:**

📦 RemindMate\_Dataset  
 ┣ 📂 Baseline  
 ┣ 📂 Multi-Move  
 ┣ 📂 Fine-Location  
 ┗ 📂 Object-Variants

## **💻 Core Code Snippets**

We open-source the core logic of our memory mechanism. These snippets illustrate how the **Encoding-Storage-Recall** pipeline and the **Hierarchical Memory Graph** are constructed.

### **Prerequisites**

* Python 3.8+  
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (for object detection)  
* LLM/VLM APIs (e.g., Qwen-Max)

### **1\. Memory Graph Structure (Static Brain)**

\# core/memory\_graph.py  
\# TODO: Release the classes for Hierarchical Memory Graph definition (V0: Flat to V4: Small object).  
\# class MemoryGraph:  
\#     ...

### **2\. Dynamic Workflow (Encoding, Storage, Recall)**

\# core/workflow.py  
\# TODO: Release the core algorithm showing how delta states are processed.  
\# def memory\_encoding(visual\_stream):  
\#     ...  
\# def memory\_storage(graph, delta\_state):  
\#     ...  
\# def memory\_recall(graph, user\_query):  
\#     ...

*(Detailed code implementation will be updated here.)*

## **📝 Citation**

If you find our work or dataset helpful, please consider citing our paper:

@article{remindmate202X,  
  title={RemindMate: An Object-Finding Companion Agent with Followable Memory},  
  author={Anonymous},  
  journal={Conference/Journal Name},  
  year={202X}  
}

## **📧 Contact**

For any questions regarding the dataset or the core logic, please open an issue or contact \[Your Email Address\].