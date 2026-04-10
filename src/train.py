import torch
import torch.nn.functional as F

def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    if hasattr(data, 'lpe') and 'GraphTransformer' in model.__class__.__name__:
        sign_flip = torch.randint(0, 2, (1, data.lpe.size(1)), device=data.lpe.device) * 2 - 1
        augmented_lpe = data.lpe * sign_flip
        out = model(data.x, data.edge_index, augmented_lpe)
    else:
        out = model(data.x, data.edge_index)
        
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(model, data, optimizer, epochs):
    for _ in range(epochs):
        train_epoch(model, data, optimizer)
        

def train_epoch_regression(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # FIX: Ensure we only pass LPE to GraphTransformers
        if hasattr(data, 'lpe') and 'GraphTransformer' in model.__class__.__name__:
            sign_flip = torch.randint(0, 2, (1, data.lpe.size(1)), device=device) * 2 - 1
            augmented_lpe = data.lpe * sign_flip
            out = model(data.x, data.edge_index, augmented_lpe, data.batch)
        else:
            out = model(data.x, data.edge_index, data.batch)
            
        loss = F.l1_loss(out, data.y.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)