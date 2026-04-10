import time
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Import configs and utilities
from src.config import DATASETS, GRAPH_DATASETS, MODELS, REGRESSION_MODELS, HYPERPARAMETERS
from src.data.loader import load_dataset, load_pcqm4mv2
from src.utils import get_model, track_gpu_memory
from src.train import train_model, train_epoch_regression
from src.evaluate import evaluate, generate_performance_plots

@torch.no_grad()
def evaluate_regression(model, loader, device):
    """Evaluates Mean Absolute Error (MAE) for Graph Regression tasks."""
    model.eval()
    total_mae = 0
    
    for data in loader:
        data = data.to(device)
        # Check if the model is a Transformer and expects Laplacian Positional Encodings
        if hasattr(data, 'lpe') and 'GraphTransformer' in model.__class__.__name__:
            out = model(data.x, data.edge_index, data.lpe, data.batch)
        else:
            out = model(data.x, data.edge_index, data.batch)
            
        mae = F.l1_loss(out, data.y.float(), reduction='sum').item()
        total_mae += mae
        
    return total_mae / len(loader.dataset)

def main():
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    # ==========================================
    # TRACK 1: NODE CLASSIFICATION (Cora, CiteSeer)
    # ==========================================
    for dataset_name in DATASETS:
        print(f"\n{'='*40}\nProcessing Classification Dataset: {dataset_name}\n{'='*40}")
        dataset, data = load_dataset(dataset_name, HYPERPARAMETERS["k_eigenvectors"])
        data = data.to(device)

        for model_name in MODELS:
            print(f"--- Training {model_name} ---")
            model = get_model(model_name, dataset.num_features, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=HYPERPARAMETERS["learning_rate"], 
                weight_decay=HYPERPARAMETERS["weight_decay"]
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            train_model(model, data, optimizer, HYPERPARAMETERS["epochs"])
            end_time = time.time()

            avg_time_per_epoch = (end_time - start_time) / HYPERPARAMETERS["epochs"]
            peak_memory = track_gpu_memory()

            acc, macro_f1, nmi = evaluate(model, data)
            print(f"Generating plots for {model_name} on {dataset_name}...")
            generate_performance_plots(model, data, dataset_name, model_name)

            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": acc,
                "Macro_F1": macro_f1,
                "NMI": nmi,
                "Validation MAE": "N/A",
                "Time/Epoch (s)": avg_time_per_epoch,
                "Peak GPU Mem (MB)": peak_memory
            })

    # ==========================================
    # TRACK 2: GRAPH REGRESSION (PCQM4Mv2)
    # ==========================================
    for dataset_name in GRAPH_DATASETS:
        print(f"\n{'='*40}\nProcessing Regression Dataset: {dataset_name}\n{'='*40}")
        
        # Load splits directly using the PyG native loader signature
        print("Loading training split...")
        train_dataset = load_pcqm4mv2(HYPERPARAMETERS["k_eigenvectors"], split='train')
        print("Loading validation split...")
        valid_dataset = load_pcqm4mv2(HYPERPARAMETERS["k_eigenvectors"], split='val')
        
        # Subset the data to avoid OOM and extreme training times on CPU
        # PCQM4Mv2 has ~3.3M train graphs, we test on 10,000 for sanity
        print("Subsetting dataset for manageable CPU execution...")
        train_subset = torch.utils.data.Subset(train_dataset, range(10000))
        valid_subset = torch.utils.data.Subset(valid_dataset, range(1000))
        
        train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_subset, batch_size=256, shuffle=False)
        
        for model_name in REGRESSION_MODELS:
            print(f"--- Training {model_name} ---")
            # We use train_dataset.num_features because Subset doesn't expose it directly
            model = get_model(model_name, train_dataset.num_features, 1).to(device) 
            optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            for epoch in range(HYPERPARAMETERS["epochs"]):
                train_loss = train_epoch_regression(model, train_loader, optimizer, device)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{HYPERPARAMETERS['epochs']} | Train MAE: {train_loss:.4f}")
            end_time = time.time()

            val_mae = evaluate_regression(model, valid_loader, device)
            print(f"Final Validation MAE: {val_mae:.4f}")
            
            avg_time_per_epoch = (end_time - start_time) / HYPERPARAMETERS["epochs"]
            peak_memory = track_gpu_memory()

            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": "N/A",
                "Macro_F1": "N/A",
                "NMI": "N/A",
                "Validation MAE": val_mae,
                "Time/Epoch (s)": avg_time_per_epoch,
                "Peak GPU Mem (MB)": peak_memory
            })

    # ==========================================
    # SAVE & DISPLAY RESULTS
    # ==========================================
    df = pd.DataFrame(results)
    print("\n" + "="*40 + "\nFINAL RESULTS\n" + "="*40)
    print(df.to_string(index=False))
    df.to_csv("results.csv", index=False)
    print("\nResults successfully saved to 'results.csv'.")

if __name__ == "__main__":
    main()