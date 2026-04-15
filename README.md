# Graph Transformer for Community Detection

This project benchmarks a graph transformer against two standard graph neural network baselines, `GCN` and `GAT`, for node-level community detection.

The current experiment suite runs on:
- `Cora`
- `CiteSeer`
- `Ego-Facebook`

For the citation datasets, the task is node classification using the dataset labels. For `Ego-Facebook`, the project builds a community-detection benchmark by generating pseudo ground-truth communities with the Louvain algorithm, then training the models to predict those community assignments.

## Project Goal

We wanted to test whether a transformer-style graph model with Laplacian positional encodings can compete with classic message-passing GNNs on community-structured graphs, while also producing useful qualitative diagnostics such as confusion matrices and t-SNE projections.

## Implemented Models

- `GCN`: graph convolution baseline from Kipf and Welling
- `GAT`: graph attention baseline
- `GraphTransformer`: transformer-based graph model built with `TransformerConv`, residual connections, layer normalization, and Laplacian positional encodings

## Current Pipeline

1. Load a dataset with normalized node features.
2. Compute Laplacian positional encodings (`k=16`) for every graph.
3. For `Ego-Facebook`, generate community labels with Louvain and create a `60/20/20` train/validation/test split.
4. Train each model for `1500` epochs using `AdamW`.
5. Evaluate on the test mask with accuracy, macro F1, and NMI.
6. Save aggregate results to `results.csv`.
7. Export a confusion matrix and t-SNE plot for every dataset/model pair into `results/plots/`.

## Repository Layout

```text
Graph-Transformer-CD/
|-- src/
|   |-- config.py
|   |-- train.py
|   |-- evaluate.py
|   |-- utils.py
|   |-- data/
|   |   `-- loader.py
|   `-- models/
|       |-- baselines.py
|       `-- transformer.py
|-- results/
|   `-- plots/
|-- docs/
|   `-- Project_Proposal___MLNS.pdf
|-- run_all.py
|-- results.csv
|-- requirements.txt
`-- README.md
```

## Installation

```bash
git clone <repository-url>
cd Graph-Transformer-CD
pip install -r requirements.txt
```

This project depends on PyTorch Geometric. If your environment does not already support it, install the matching PyG wheels for your local PyTorch and CUDA/CPU setup before running experiments.

## Running Experiments

Run the full benchmark:

```bash
python run_all.py
```

This will:
- train `GCN`, `GAT`, and `GraphTransformer`
- process `Cora`, `CiteSeer`, and `Ego-Facebook`
- overwrite `results.csv`
- generate plots in `results/plots/`

## Default Configuration

The experiment defaults in `src/config.py` are:

```python
DATASETS = ["Cora", "CiteSeer", "Ego-Facebook"]
MODELS = ["GCN", "GAT", "GraphTransformer"]

HYPERPARAMETERS = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "epochs": 1500,
    "k_eigenvectors": 16,
    "hidden_dim": 64,
    "num_layers": 2,
    "num_heads": 4
}
```

## Dataset Notes

### Cora and CiteSeer

- Loaded with `torch_geometric.datasets.Planetoid`
- Use the standard node labels provided by the datasets
- Feature normalization is applied during loading

### Ego-Facebook

- Loaded with `torch_geometric.datasets.SNAPDataset`
- Community labels are generated with NetworkX Louvain clustering
- Random train/validation/test masks are created inside the loader
- If node features are missing, node degree is used as a fallback feature

This makes `Ego-Facebook` a structurally driven benchmark rather than a native labeled node-classification dataset.

## Evaluation Metrics

- `Accuracy`: percentage of correct node predictions on the test split
- `Macro_F1`: class-balanced F1 score
- `NMI`: normalized mutual information between predicted and target communities
- `Time/Epoch (s)`: average training time per epoch
- `Peak GPU Mem (MB)`: peak allocated GPU memory; this is `0.0` on CPU runs

## Current Results

The latest committed results in `results.csv` are:

| Dataset | Model | Accuracy | Macro F1 | NMI | Time / Epoch (s) | Peak GPU Mem (MB) |
|---|---|---:|---:|---:|---:|---:|
| Cora | GCN | 0.7850 | 0.7791 | 0.5815 | 0.0306 | 0.0 |
| Cora | GAT | 0.7810 | 0.7785 | 0.5634 | 0.1300 | 0.0 |
| Cora | GraphTransformer | 0.5060 | 0.4656 | 0.2381 | 0.0965 | 0.0 |
| CiteSeer | GCN | 0.6900 | 0.6533 | 0.4267 | 0.0522 | 0.0 |
| CiteSeer | GAT | 0.6700 | 0.6454 | 0.3945 | 0.1882 | 0.0 |
| CiteSeer | GraphTransformer | 0.5120 | 0.4927 | 0.2226 | 0.1148 | 0.0 |
| Ego-Facebook | GCN | 0.5714 | 0.2668 | 0.3834 | 0.0115 | 0.0 |
| Ego-Facebook | GAT | 0.7571 | 0.5734 | 0.6418 | 0.0355 | 0.0 |
| Ego-Facebook | GraphTransformer | 0.8571 | 0.8804 | 0.7498 | 0.0409 | 0.0 |

## Result Takeaways

- On `Cora` and `CiteSeer`, the classic baselines outperform the graph transformer in this configuration.
- On `Ego-Facebook`, the `GraphTransformer` performs best across accuracy, macro F1, and NMI.
- The transformer appears to benefit more on the structurally defined community task than on the citation benchmarks.

## Generated Artifacts

For every dataset/model pair, the project saves:
- a confusion matrix PNG
- a t-SNE visualization of node embeddings

Examples already included in the repo:
- `results/plots/Cora_GCN_confusion_matrix.png`
- `results/plots/Cora_GraphTransformer_tsne.png`
- `results/plots/Ego-Facebook_GAT_confusion_matrix.png`
- `results/plots/Ego-Facebook_GraphTransformer_tsne.png`

## References

- Dwivedi, V. P., and Bresson, X. "A Generalization of Transformer Networks to Graphs."
- Kipf, T. N., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks."
- Velickovic, P., et al. "Graph Attention Networks."
