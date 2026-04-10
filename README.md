# Graph Transformer for Community Detection

This project compares the performance of Graph Transformers with traditional Graph Neural Networks (GNNs) like GCN and GAT for community detection tasks on graph datasets.

## What is Community Detection?

Community detection is the process of identifying groups (communities) of nodes in a network that are more densely connected to each other than to the rest of the network. This is useful in social networks, citation networks, and other graph-structured data to understand underlying structures and relationships.

## Project Overview

This project implements and evaluates:
- **Graph Transformer**: A lightweight transformer-based model adapted for graphs, using Laplacian Positional Encodings and local attention mechanisms for efficiency.
- **GCN (Graph Convolutional Network)**: A traditional message-passing GNN.
- **GAT (Graph Attention Network)**: An attention-based GNN that weighs neighbor importance.

The models are tested on citation network datasets where:
- Nodes represent academic papers
- Edges represent citations between papers
- Node labels represent research topics (communities)

## Features

- Implementation of Graph Transformer with O(E) complexity (efficient for large graphs)
- Laplacian Positional Encodings for structural information
- Comparison with baseline GNN models
- Evaluation on Cora and CiteSeer datasets
- Performance metrics: Accuracy, Macro F1, Normalized Mutual Information (NMI)
- GPU memory tracking and timing analysis

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Graph-Transformer-CD
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: This project requires PyTorch Geometric. If you encounter installation issues, refer to the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage

Run all experiments:
```bash
python run_all.py
```

This will train and evaluate all models on the Cora and CiteSeer datasets. Results will be saved to `results.csv` and plots will be generated in the `results/plots/` directory.

## Datasets

- **Cora**: A citation network with 2,708 nodes, 5,429 edges, and 7 classes
- **CiteSeer**: A citation network with 3,312 nodes, 4,723 edges, and 6 classes

## Results

The project evaluates models based on:
- **Accuracy**: Percentage of correctly classified nodes
- **Macro F1**: Balanced measure of precision and recall across classes
- **NMI**: Normalized Mutual Information measuring clustering quality
- **Training Time**: Time per epoch
- **GPU Memory Usage**: Peak memory consumption

Example results (may vary based on hyperparameters):
- GCN and GAT typically achieve higher accuracy (70-80%) compared to Graph Transformer
- Graph Transformer shows competitive performance with potentially better scalability for larger graphs

## Project Structure

```
├── src/
│   ├── config.py          # Hyperparameters and model configurations
│   ├── data/
│   │   └── loader.py      # Dataset loading utilities
│   ├── models/
│   │   ├── baselines.py   # GCN and GAT implementations
│   │   └── transformer.py # Graph Transformer implementation
│   ├── train.py           # Training functions
│   ├── evaluate.py        # Evaluation and plotting
│   └── utils.py           # Helper functions
├── data/                  # Dataset storage
├── results/               # Output results and plots
├── run_all.py             # Main execution script
├── requirements.txt       # Python dependencies
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

## References

- Dwivedi, V. P., & Bresson, X. (2021). A Generalization of Transformer Networks to Graphs. arXiv preprint arXiv:2012.09699.
- Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
- Veličković, P., et al. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.
