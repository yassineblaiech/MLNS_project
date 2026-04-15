import torch
import numpy as np
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, SNAPDataset
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_networkx
from scipy.sparse.linalg import eigs

def compute_lpe(edge_index, num_nodes, k):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    
    try:
        if num_nodes <= k + 1:
            L_dense = L.toarray()
            evals, evecs = np.linalg.eigh(L_dense)
            evecs = evecs[:, 1:k+1] 
        else:
            evals, evecs = eigs(L, k=k+1, which='SM')
            evecs = np.real(evecs[:, 1:])
    except:
        evecs = np.zeros((num_nodes, 0))
    
    pad_size = k - evecs.shape[1]
    if pad_size > 0:
        evecs = np.pad(evecs, ((0, 0), (0, pad_size)), mode='constant')
        
    return torch.from_numpy(evecs).float()

class SafeLaplacianPETransform:
    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        data.lpe = compute_lpe(data.edge_index, data.num_nodes, self.k)
        return data

def load_dataset(name, k_eigenvectors):
    transform = T.NormalizeFeatures()
    if name in ["Cora", "CiteSeer"]:
        dataset = Planetoid(root=f'data/{name}', name=name, transform=transform)
        data = dataset[0]
    elif name == "Ego-Facebook":
        dataset = SNAPDataset(root='data/Ego-Facebook', name='ego-facebook', transform=transform)
        data = dataset[0]
        num_nodes = data.num_nodes
        
        # 1. Generate structural communities using Louvain
        print("Running Louvain algorithm to generate ground-truth communities...")
        G = to_networkx(data, to_undirected=True)
        communities = nx.community.louvain_communities(G, seed=42)
        
        # 2. Map Louvain communities to a 1D label tensor
        data.y = torch.zeros(num_nodes, dtype=torch.long)
        for comm_idx, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                data.y[node] = comm_idx
                
        print(f"Generated {len(communities)} distinct communities.")

        # 3. Create Train/Val/Test masks
        indices = torch.randperm(num_nodes)
        train_idx = indices[:int(0.6 * num_nodes)]
        val_idx = indices[int(0.6 * num_nodes):int(0.8 * num_nodes)]
        test_idx = indices[int(0.8 * num_nodes):]
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        # 4. Handle missing features (Ego-Facebook nodes don't always have rich features)
        if getattr(data, 'x', None) is None:
            # Option A: Use a simple constant feature
            # data.x = torch.ones((num_nodes, 1))
            
            # Option B: Use node degrees as features (Better for structural tasks)
            degrees = torch.tensor([val for (node, val) in G.degree()], dtype=torch.float).view(-1, 1)
            data.x = degrees
            
    data.lpe = compute_lpe(data.edge_index, data.num_nodes, k_eigenvectors)
    return dataset, data