import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import umap.umap_ as umap
from tqdm import tqdm

def load_embeddings(base_dir, class_name):
    """Load embeddings and filenames for a specific class."""
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        return [], []
    
    embeddings = []
    filenames = []
    
    for emb_file in tqdm(os.listdir(class_dir), desc=f"Loading {class_name} embeddings"):
        if emb_file.endswith('_embeddings.npy'):
            file_path = os.path.join(class_dir, emb_file)
            emb = np.load(file_path)
            embeddings.append(emb)
            # Get the base filename without the _embeddings.npy suffix
            base_name = os.path.splitext(emb_file)[0].replace('_embeddings', '')
            filenames.append(f"{base_name}")
    
    return embeddings, filenames

def apply_pooling(embeddings, pool_type='mean'):
    """Apply mean or max pooling to time-series embeddings."""
    if pool_type == 'mean':
        return np.array([np.mean(emb, axis=0) for emb in embeddings])
    elif pool_type == 'max':
        return np.array([np.max(emb, axis=0) for emb in embeddings])
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")

def main():
    # Set up paths
    base_dir = "sliding_window_embeddings"
    output_dir = "umap_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process embeddings
    all_embeddings = []
    all_filenames = []
    all_labels = []
    
    # Process positive class
    pos_embeddings, pos_filenames = load_embeddings(base_dir, "positive")
    if pos_embeddings:
        pos_pooled = apply_pooling(pos_embeddings, 'max')  # Using max pooling as in your example
        all_embeddings.extend(pos_pooled)
        all_filenames.extend(pos_filenames)
        all_labels.extend([1] * len(pos_pooled))
    
    # Process negative class
    neg_embeddings, neg_filenames = load_embeddings(base_dir, "negative")
    if neg_embeddings:
        neg_pooled = apply_pooling(neg_embeddings, 'max')  # Using max pooling as in your example
        all_embeddings.extend(neg_pooled)
        all_filenames.extend(neg_filenames)
        all_labels.extend([0] * len(neg_pooled))
    
    if not all_embeddings:
        print("No embeddings found!")
        return
    
    # Convert to numpy arrays
    X = np.array(all_embeddings)
    y = np.array(all_labels)
    
    # Apply UMAP
    print("\nApplying UMAP...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1, n_components=2)
    umap_results = reducer.fit_transform(X)
    
    # Find points above y=4
    y_threshold = 4
    above_threshold = umap_results[:, 1] > y_threshold
    
    # Print information about points above y=4
    print(f"\nFound {sum(above_threshold)} points above y={y_threshold}:")
    print("-" * 80)
    
    for i, (point, is_above) in enumerate(zip(umap_results, above_threshold)):
        if is_above:
            print(f"File: {all_filenames[i]}")
            print(f"  Class: {'Positive' if y[i] == 1 else 'Negative'}")
            print(f"  UMAP Coordinates: ({point[0]:.4f}, {point[1]:.4f})")
            print("-" * 80)
    
    # Plot the points
    plt.figure(figsize=(12, 10))
    
    # Plot all points
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], 
                         c=y, cmap='viridis', alpha=0.6, 
                         edgecolor='w', s=50)
    
    # Highlight points above threshold
    plt.scatter(umap_results[above_threshold, 0], umap_results[above_threshold, 1],
                c='red', edgecolor='black', s=100, 
                label=f'Above y={y_threshold}')
    
    # Add a horizontal line at y=4
    plt.axhline(y=y_threshold, color='r', linestyle='--', alpha=0.5)
    plt.text(plt.xlim()[1], y_threshold, f'y = {y_threshold}', 
             va='bottom', ha='right', color='r')
    
    plt.title(f'UMAP of Audio Embeddings\n(Points above y={y_threshold} highlighted in red)', fontsize=14)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.colorbar(scatter, label='Class (1=Positive, 0=Negative)')
    
    # Save the plot
    output_path = os.path.join(output_dir, f'umap_above_{y_threshold}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nSaved UMAP plot to {output_path}")
    
    # Save the points above threshold to a file
    above_indices = np.where(above_threshold)[0]
    above_data = [{
        'filename': all_filenames[i],
        'class': 'positive' if y[i] == 1 else 'negative',
        'umap_x': float(umap_results[i, 0]),
        'umap_y': float(umap_results[i, 1])
    } for i in above_indices]
    
    output_json = os.path.join(output_dir, f'points_above_{y_threshold}.json')
    with open(output_json, 'w') as f:
        json.dump(above_data, f, indent=2)
    
    print(f"Saved points above y={y_threshold} to {output_json}")

if __name__ == "__main__":
    main()
