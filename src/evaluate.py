import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, normalized_mutual_info_score, confusion_matrix
from sklearn.manifold import TSNE

# Ensure the output directory exists
os.makedirs('results/plots', exist_ok=True)

@torch.no_grad()
def get_predictions(model, data):
    """Helper function to extract logits and predictions."""
    model.eval()
    if hasattr(data, 'lpe') and 'GraphTransformer' in model.__class__.__name__:
        out = model(data.x, data.edge_index, data.lpe)
    else:
        out = model(data.x, data.edge_index)
    
    pred = out.argmax(dim=1)
    return out, pred

@torch.no_grad()
def evaluate(model, data):
    """Calculates quantitative metrics."""
    out, pred = get_predictions(model, data)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

    acc = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return acc, macro_f1, nmi

def plot_confusion_matrix(y_true, y_pred, dataset_name, model_name):
    """Plots and saves a heatmap of the test set confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.xlabel('Predicted Community')
    plt.ylabel('True Community')
    plt.tight_layout()
    plt.savefig(f'results/plots/{dataset_name}_{model_name}_confusion_matrix.png', dpi=300)
    plt.close()

def plot_tsne_embeddings(embeddings, labels, dataset_name, model_name):
    """Reduces embeddings to 2D using t-SNE and plots them colored by community."""
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.legend(*scatter.legend_elements(), title="Communities", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE Node Embeddings - {model_name} on {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'results/plots/{dataset_name}_{model_name}_tsne.png', dpi=300)
    plt.close()

def generate_performance_plots(model, data, dataset_name, model_name):
    """Orchestrator to generate all diagnostic plots for a given model."""
    out, pred = get_predictions(model, data)
    
    # 1. t-SNE plot (using the entire graph to see full structural clustering)
    embeddings = out.cpu().numpy()
    labels = data.y.cpu().numpy()
    plot_tsne_embeddings(embeddings, labels, dataset_name, model_name)
    
    # 2. Confusion Matrix (using only the test set to evaluate true generalization)
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    plot_confusion_matrix(y_true, y_pred, dataset_name, model_name)