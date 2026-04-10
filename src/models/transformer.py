import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, k_eigenvectors, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        # 1. Deeper processing for LPE
        self.pe_mlp = torch.nn.Sequential(
            torch.nn.Linear(k_eigenvectors, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.node_emb = torch.nn.Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, beta=True, dropout=dropout))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, lpe):
        # 2. Process LPE and add to features
        pe = self.pe_mlp(lpe)
        x = self.node_emb(x) + pe
        
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = norm(x) 
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res # Residual connection
            
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class GraphTransformerRegression(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, k_eigenvectors, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        self.pe_mlp = torch.nn.Sequential(
            torch.nn.Linear(k_eigenvectors, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.node_emb = torch.nn.Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, beta=True, dropout=dropout))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, lpe, batch):
        pe = self.pe_mlp(lpe)
        x = self.node_emb(x.float()) + pe
        
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = norm(x) 
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res
            
        x_graph = global_mean_pool(x, batch)
        out = self.mlp(x_graph)
        return out.squeeze()