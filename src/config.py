DATASETS = ["Cora", "CiteSeer"]
GRAPH_DATASETS = ["PCQM4Mv2"]

MODELS = ["GCN", "GAT", "GraphTransformer"]
REGRESSION_MODELS = ["GCNRegression", "GATRegression", "GraphTransformerRegression"]

HYPERPARAMETERS = {
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "epochs": 400,
    "k_eigenvectors": 16,
    "hidden_dim": 64,
    "num_layers": 2,
    "num_heads": 4
}