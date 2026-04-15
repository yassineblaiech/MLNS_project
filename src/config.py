DATASETS = ["Cora", "CiteSeer", "Ego-Facebook"]
GRAPH_DATASETS = []

MODELS = ["GCN", "GAT", "GraphTransformer"]
REGRESSION_MODELS = []

HYPERPARAMETERS = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "epochs": 1500,
    "k_eigenvectors": 16,
    "hidden_dim": 64,
    "num_layers": 2,
    "num_heads": 4
}