import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PCQM4Mv2
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs

def compute_lpe(edge_index, num_nodes, k):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    
    try:
        # If the graph is very small, use standard dense eigenvalue decomposition to avoid warnings
        if num_nodes <= k + 1:
            L_dense = L.toarray()
            evals, evecs = np.linalg.eigh(L_dense)
            # Take up to k non-trivial eigenvectors
            evecs = evecs[:, 1:k+1] 
        else:
            # Use sparse solver for larger graphs
            evals, evecs = eigs(L, k=k+1, which='SM')
            evecs = np.real(evecs[:, 1:])
    except:
        evecs = np.zeros((num_nodes, 0))
    
    # Pad the remaining dimensions with zeros so the shape is ALWAYS (num_nodes, k)
    pad_size = k - evecs.shape[1]
    if pad_size > 0:
        evecs = np.pad(evecs, ((0, 0), (0, pad_size)), mode='constant')
        
    return torch.from_numpy(evecs).float()

class SafeLaplacianPETransform:
    """Custom transform to safely pad LPEs for graphs with fewer than k nodes."""
    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        data.lpe = compute_lpe(data.edge_index, data.num_nodes, self.k)
        return data

def load_dataset(name, k_eigenvectors):
    transform = T.NormalizeFeatures()
    dataset = Planetoid(root=f'data/{name}', name=name, transform=transform)
    data = dataset[0]
    data.lpe = compute_lpe(data.edge_index, data.num_nodes, k_eigenvectors)
    return dataset, data

def load_pcqm4mv2(k_eigenvectors, split='train'):
    transform = SafeLaplacianPETransform(k=k_eigenvectors)
    dataset = PCQM4Mv2(root='data/PCQM4Mv2', split=split, transform=transform)
    return dataset