import time
import torch
import pandas as pd
from src.config import DATASETS, MODELS, HYPERPARAMETERS
from src.data.loader import load_dataset
from src.utils import get_model, track_gpu_memory
from src.train import train_model
from src.evaluate import evaluate, generate_performance_plots

def main():
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    for dataset_name in DATASETS:
        print(f"\n{'='*40}\nProcessing Classification Dataset: {dataset_name}\n{'='*40}")
        dataset, data = load_dataset(dataset_name, HYPERPARAMETERS["k_eigenvectors"])
        data = data.to(device)

        num_features = data.x.size(1) if data.x is not None else 1
        num_classes = len(torch.unique(data.y)) if data.y is not None else 1

        for model_name in MODELS:
            print(f"--- Training {model_name} ---")
            model = get_model(model_name, num_features, num_classes).to(device)
            optimizer = torch.optim.AdamW(
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

    df = pd.DataFrame(results)
    print("\n" + "="*40 + "\nFINAL RESULTS\n" + "="*40)
    print(df.to_string(index=False))
    df.to_csv("results.csv", index=False)
    print("\nResults successfully saved to 'results.csv'.")

if __name__ == "__main__":
    main()