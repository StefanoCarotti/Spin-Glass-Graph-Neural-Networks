# AI Model for Physics Project

## Task 1: Graph Neural Network (GNN) for Spin Glass Systems
## Task 2: Comparing the performance of GNN and MLPs

### Project Overview
The goal of this project is to develop a Graph Neural Network (GNN) to study **spin glass systems**. A spin glass is a magnetic system represented as a graph, where:
- **Nodes** represent spins (which can take values of +1 or -1),
- **Edges** represent interactions between spins,
- The **energy** of the system is derived from the Hamiltonian, and
- The **magnetization** is the sum of the spins.

The project is divided into two main tasks:
1. Build a GNN to predict the energy of the spin glass system and analyze how learning changes with graph connectivity and magnetic field.
2. Compare GNNs to Multi-Layer Perceptrons (MLPs) for learning both energy and magnetization of the system.

This repository contains both the **Task 1** and **Task 2** implementation.

---

## Dataset Construction

We create datasets of spin glass systems where:
- Each graph has **L nodes** with random spins and random edge weights,
- The dataset includes various configurations of graph connectivity, with external magnetic fields (denoted as `h`).

The code constructs spin glass graphs for different numbers of edges (e.g., open chains and fully connected graphs) and trains a GNN to predict their energy. A brief description of the dataset creation process is as follows:

1. **Graph Initialization**: A graph is initialized with random spins and edge weights.
2. **Energy Calculation**: The energy of the system is computed based on spins and interactions.
3. **Dataset Generation**: Datasets are created for training the GNN, where each graph is represented in a format compatible with PyTorch Geometric.

---

## GNN Model

The GNN model uses graph convolutional layers to extract features from the spin glass system. Key components include:
- **Graph Convolution Layers**: 5 layers to process node and edge information,
- **Fully Connected Layers**: To predict the final energy value,
- **Dropout Layer**: To prevent overfitting.

### Model Architecture:
- **Input**: Spin values of nodes and edge weights,
- **Output**: Predicted energy of the system.
