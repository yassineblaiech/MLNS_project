import torch
from src.models.baselines import GCN, GAT, GCNRegression, GATRegression
from src.models.transformer import GraphTransformer, GraphTransformerRegression
from src.config import HYPERPARAMETERS

def get_model(model_name, in_channels, out_channels):
    if model_name == "GCN":
        return GCN(in_channels, HYPERPARAMETERS["hidden_dim"], out_channels, HYPERPARAMETERS["num_layers"])
    elif model_name == "GAT":
        return GAT(in_channels, HYPERPARAMETERS["hidden_dim"], out_channels, HYPERPARAMETERS["num_layers"], HYPERPARAMETERS["num_heads"])
    elif model_name == "GraphTransformer":
        return GraphTransformer(in_channels, HYPERPARAMETERS["hidden_dim"], out_channels, HYPERPARAMETERS["num_layers"], HYPERPARAMETERS["num_heads"], HYPERPARAMETERS["k_eigenvectors"])
    elif model_name == "GCNRegression":
        return GCNRegression(in_channels, HYPERPARAMETERS["hidden_dim"], out_channels, HYPERPARAMETERS["num_layers"])
    elif model_name == "GATRegression":
        return GATRegression(in_channels, HYPERPARAMETERS["hidden_dim"], out_channels, HYPERPARAMETERS["num_layers"], HYPERPARAMETERS["num_heads"])
    elif model_name == "GraphTransformerRegression":
        return GraphTransformerRegression(in_channels, HYPERPARAMETERS["hidden_dim"], out_channels, HYPERPARAMETERS["num_layers"], HYPERPARAMETERS["num_heads"], HYPERPARAMETERS["k_eigenvectors"])
    else:
        raise ValueError("Unknown model")

def track_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0