import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
import umap.umap_ as umap
import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8')  # Use seaborn style with version specification
sns.set_theme(style="whitegrid")  # Use seaborn's whitegrid theme
sns.set_palette('colorblind')  # Use colorblind-friendly palette

def load_embeddings(base_dir, label, max_files=100, target_length=128):
    """Load and normalize embeddings from the specified directory."""
    embeddings = []
    
    # Get list of all embedding files
    embed_dir = Path(base_dir) / label
    embed_files = sorted(embed_dir.glob("*.npy"))[:max_files]
    
    for embed_file in tqdm(embed_files, desc=f"Loading {label} embeddings"):
        try:
            # Load embedding
            embedding = np.load(embed_file)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing {embed_file}: {e}")
            continue
    
    if not embeddings:
        raise ValueError(f"No valid embeddings found in {embed_dir}")
        
    return (embeddings)

def apply_pooling(embeddings, pool_type='mean'):
    """Apply mean or max pooling to time-series embeddings."""
    if pool_type == 'mean':
        return np.mean(embeddings, axis=1)  # Mean across time dimension
    elif pool_type == 'max':
        return np.max(embeddings, axis=1)  # Max across time dimension
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")

def plot_pca(embeddings, labels, title, save_path=None):
    """Generate and save PCA plot with centroids in 2D and 3D."""
    # 2D PCA
    pca_2d = PCA(n_components=2)
    pca_2d_results = pca_2d.fit_transform(embeddings)
    
    # 3D PCA
    pca_3d = PCA(n_components=3)
    pca_3d_results = pca_3d.fit_transform(embeddings)
    
    # Save 2D plot
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_2d_results[:, 0], pca_2d_results[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6, 
                            edgecolor='w', s=50)
        
        # Add centroids to 2D PCA
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(pca_2d_results[mask], axis=0)
            plt.scatter(centroid[0], centroid[1], 
                       c='red' if label == 1 else 'blue', 
                       marker='X', s=200, edgecolor='black',
                       linewidth=1.5, label=f'Class {int(label)} Centroid')
        
        plt.title(f"PCA: {title}\n(Blue=Negative, Yellow=Positive, ★=Centroid)", fontsize=14)
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, label='Class (1=Positive, 0=Negative)')
        plt.legend()
        plt.savefig(f"{base_path}_2d.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save 3D PCA
        plot_3d(pca_3d_results, labels,
               f"PCA: {title}",
               f"{base_path}_3d.png")
    else:
        plot_2d_with_centroids(pca_2d_results, labels, 
                              f"PCA: {title}")
        plot_3d(pca_3d_results, labels, 
               f"PCA: {title}")
    
    return pca_2d_results

def plot_2d_with_centroids(points, labels, title, save_path=None):
    """Plot 2D points with class centroids."""
    plt.figure(figsize=(10, 8))
    
    # Plot points
    scatter = plt.scatter(points[:, 0], points[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6, 
                         edgecolor='w', s=50)
    
    # Calculate and plot centroids
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        centroid = np.mean(points[mask], axis=0)
        plt.scatter(centroid[0], centroid[1], 
                   c='red' if label == 1 else 'blue', 
                   marker='X', s=200, edgecolor='black',
                   linewidth=1.5, label=f'Class {int(label)} Centroid')
    
    plt.title(f"{title}\n(Blue=Negative, Yellow=Positive, ★=Centroid)", fontsize=14)
    plt.colorbar(scatter, label='Class (1=Positive, 0=Negative)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_3d(points, labels, title, save_path=None):
    """Generate 3D visualization of the data."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=labels, cmap='viridis', alpha=0.7,
                        edgecolor='w', s=50)
    
    # Calculate and plot centroids
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        centroid = np.mean(points[mask], axis=0)
        ax.scatter(centroid[0], centroid[1], centroid[2],
                  c='red' if label == 1 else 'blue',
                  marker='*', s=300, edgecolor='black',
                  linewidth=1.5, label=f'Class {int(label)} Centroid')
    
    ax.set_title(f"3D {title}\n(Blue=Negative, Yellow=Positive, ★=Centroid)", fontsize=14)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_tsne(embeddings, labels, title, save_path=None):
    """Generate and save t-SNE plot with centroids."""
    # 2D t-SNE
    tsne_2d = TSNE(n_components=2, random_state=42)
    tsne_2d_results = tsne_2d.fit_transform(embeddings)
    
    # 3D t-SNE
    tsne_3d = TSNE(n_components=3, random_state=42)
    tsne_3d_results = tsne_3d.fit_transform(embeddings)
    
    # Save 2D plot
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        plot_2d_with_centroids(tsne_2d_results, labels, 
                              f"t-SNE: {title}", 
                              f"{base_path}_2d.png")
        
        # Save 3D plot
        plot_3d(tsne_3d_results, labels,
               f"t-SNE: {title}",
               f"{base_path}_3d.png")
    else:
        plot_2d_with_centroids(tsne_2d_results, labels, f"t-SNE: {title}")
        plot_3d(tsne_3d_results, labels, f"t-SNE: {title}")
    
    return tsne_2d_results

def plot_umap(embeddings, labels, title, save_path=None):
    """Generate and save UMAP plot with centroids in 2D and 3D."""
    # 2D UMAP
    reducer_2d = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1, n_components=2)
    umap_2d_results = reducer_2d.fit_transform(embeddings)
    
    # 3D UMAP
    reducer_3d = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1, n_components=3)
    umap_3d_results = reducer_3d.fit_transform(embeddings)
    
    # Save 2D plot
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        plot_2d_with_centroids(umap_2d_results, labels, 
                              f"UMAP: {title}", 
                              f"{base_path}_2d.png")
        
        # Save 3D plot
        plot_3d(umap_3d_results, labels,
               f"UMAP: {title}",
               f"{base_path}_3d.png")
    else:
        plot_2d_with_centroids(umap_2d_results, labels, f"UMAP: {title}")
        plot_3d(umap_3d_results, labels, f"UMAP: {title}")
    
    return umap_2d_results

def analyze_boundary_cases(embeddings, labels, filenames, title, threshold_percentile=10):
    """Analyze samples near the decision boundary from both classes."""
    from sklearn.neighbors import NearestNeighbors
    
    # Separate positive and negative samples
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    # Calculate centroids
    pos_centroid = np.mean(embeddings[pos_idx], axis=0)
    neg_centroid = np.mean(embeddings[neg_idx], axis=0)
    
    # Calculate distances to centroids
    pos_distances = np.linalg.norm(embeddings - pos_centroid, axis=1)
    neg_distances = np.linalg.norm(embeddings - neg_centroid, axis=1)
    
    # 1. Find positive samples close to negative centroid
    pos_to_neg_dist = neg_distances[labels == 1]
    pos_threshold = np.percentile(pos_to_neg_dist, threshold_percentile)
    pos_boundary_indices = pos_idx[pos_to_neg_dist <= pos_threshold]
    
    # 2. Find negative samples close to positive centroid
    neg_to_pos_dist = pos_distances[labels == 0]
    neg_threshold = np.percentile(neg_to_pos_dist, threshold_percentile)
    neg_boundary_indices = neg_idx[neg_to_pos_dist <= neg_threshold]
    
    print(f"\nAnalyzing {len(pos_boundary_indices)} positive samples near negative class:")
    print("-" * 80)
    
    # Get nearest neighbors in negative class for positive boundary cases
    neg_nbrs = NearestNeighbors(n_neighbors=3).fit(embeddings[neg_idx])
    
    for idx in pos_boundary_indices:
        sample = embeddings[idx]
        filename = filenames[idx]
        
        # Find nearest negative samples
        distances, nn_indices = neg_nbrs.kneighbors([sample])
        
        print(f"\nPositive Sample Near Negative Class: {filename}")
        print(f"Distance to negative centroid: {neg_distances[idx]:.4f}")
        print(f"Distance to positive centroid: {pos_distances[idx]:.4f}")
        print(f"Nearest negative samples:")
        
        for i, (dist, nn_idx) in enumerate(zip(distances[0], nn_indices[0])):
            print(f"  {i+1}. {filenames[neg_idx[nn_idx]]} (distance: {dist:.4f})")
    
    print(f"\n\nAnalyzing {len(neg_boundary_indices)} negative samples near positive class:")
    print("-" * 80)
    
    # Get nearest neighbors in positive class for negative boundary cases
    pos_nbrs = NearestNeighbors(n_neighbors=3).fit(embeddings[pos_idx])
    
    for idx in neg_boundary_indices:
        sample = embeddings[idx]
        filename = filenames[idx]
        
        # Find nearest positive samples
        distances, nn_indices = pos_nbrs.kneighbors([sample])
        
        print(f"\nNegative Sample Near Positive Class: {filename}")
        print(f"Distance to positive centroid: {pos_distances[idx]:.4f}")
        print(f"Distance to negative centroid: {neg_distances[idx]:.4f}")
        print(f"Nearest positive samples:")
        
        for i, (dist, nn_idx) in enumerate(zip(distances[0], nn_indices[0])):
            print(f"  {i+1}. {filenames[pos_idx[nn_idx]]} (distance: {dist:.4f})")
    
    return pos_boundary_indices, neg_boundary_indices


def plot_distance_distributions(embeddings, labels, title, save_path=None):
    """Plot KDE of within-class and between-class distances."""
    # Calculate pairwise distances
    distances = squareform(pdist(embeddings, 'euclidean'))
    
    # Get indices for within-class and between-class pairs
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    # Calculate within-class distances
    pos_distances = distances[np.ix_(pos_idx, pos_idx)][np.triu_indices(len(pos_idx), k=1)]
    neg_distances = distances[np.ix_(neg_idx, neg_idx)][np.triu_indices(len(neg_idx), k=1)]
    within_distances = np.concatenate([pos_distances, neg_distances])
    
    # Calculate between-class distances
    between_distances = distances[np.ix_(pos_idx, neg_idx)].flatten()
    
    # Create KDE plots
    plt.figure(figsize=(12, 6))
    
    # Plot KDE for within-class distances
    kde_within = gaussian_kde(within_distances)
    x = np.linspace(min(within_distances.min(), between_distances.min()),
                   max(within_distances.max(), between_distances.max()), 1000)
    plt.plot(x, kde_within(x), label='Within-class distances', linewidth=2)
    
    # Plot KDE for between-class distances
    kde_between = gaussian_kde(between_distances)
    plt.plot(x, kde_between(x), label='Between-class distances', linewidth=2)
    
    plt.title(f'{title}\nDistance Distribution Analysis', fontsize=14)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def evaluate_clustering(embeddings, labels, pool_type):
    """Evaluate clustering quality and return metrics."""
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    # Reduce dimensions for faster clustering
    pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca)
    
    # Calculate metrics
    metrics = {
        'pooling_type': pool_type,
        'kmeans_silhouette': silhouette_score(X_pca, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else None,
        'kmeans_calinski': calinski_harabasz_score(X_pca, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else None,
        'kmeans_davies': davies_bouldin_score(X_pca, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else None,
        'dbscan_silhouette': silhouette_score(X_pca, dbscan_labels) if len(np.unique(dbscan_labels)) > 1 else None,
        'dbscan_calinski': calinski_harabasz_score(X_pca, dbscan_labels) if len(np.unique(dbscan_labels)) > 1 else None,
        'dbscan_davies': davies_bouldin_score(X_pca, dbscan_labels) if len(np.unique(dbscan_labels)) > 1 else None,
    }
    
    return metrics

def load_embeddings_with_filenames(base_dir, label, max_files=100):
    """Load embeddings with their corresponding filenames."""
    embeddings = []
    filenames = []
    
    # Get all embedding files
    class_dir = os.path.join(base_dir, label.lower())
    if not os.path.exists(class_dir):
        return np.array([]), []
        
    files = [f for f in os.listdir(class_dir) if f.endswith('_embeddings.npy')]
    
    # Limit number of files if needed
    if max_files and len(files) > max_files:
        files = files[:max_files]
    
    # Load each embedding file
    for f in tqdm(files, desc=f"Loading {label} embeddings"):
        try:
            emb_path = os.path.join(class_dir, f)
            emb = np.load(emb_path)
            
            # Handle different embedding shapes
            if len(emb.shape) == 1:  # Already pooled
                pooled = emb
            elif len(emb.shape) == 2:  # Time series of embeddings
                pooled = np.mean(emb, axis=0)  # Default to mean pooling
            else:
                continue
                
            embeddings.append(pooled)
            filenames.append(f.replace('_embeddings.npy', ''))
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    return np.array(embeddings), filenames


def main():
    """Main function to analyze pooled embeddings."""
    # Set up paths
    base_dir = "sliding_window_embeddings"
    output_dir = "pooling_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = []
    all_filenames = []
    
    # Process each pooling type
    for pool_type in ['mean', 'max']:
        print(f"\nProcessing {pool_type} pooling...")
        
        # Load and pool positive embeddings with filenames
        print("Loading and pooling positive embeddings...")
        pos_embeddings, pos_filenames = load_embeddings_with_filenames(base_dir, "positive")
        pos_pooled = [np.mean(emb, axis=0) if len(emb.shape) > 1 else emb 
                     for emb in pos_embeddings]
        
        # Load and pool negative embeddings with filenames
        print("Loading and pooling negative embeddings...")
        neg_embeddings, neg_filenames = load_embeddings_with_filenames(base_dir, "negative")
        neg_pooled = [np.mean(emb, axis=0) if len(emb.shape) > 1 else emb 
                     for emb in neg_embeddings]
        
        # Combine filenames
        all_filenames = pos_filenames + neg_filenames
        
        print(f"Loaded {len(pos_pooled)} positive and {len(neg_pooled)} negative pooled embeddings")
        
        # Create labels (1 for positive, 0 for negative)
        pos_labels = np.ones(len(pos_pooled))
        neg_labels = np.zeros(len(neg_pooled))
        
        # Combine and stack
        all_embeddings = np.vstack(pos_pooled + neg_pooled)
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        # Analyze boundary cases for both positive and negative samples
        if len(pos_pooled) > 0 and len(neg_pooled) > 0:
            pos_boundary, neg_boundary = analyze_boundary_cases(all_embeddings, all_labels, all_filenames, 
                                                              f"{pool_type.capitalize()} Pooling")
            
            # Save boundary cases for further analysis
            boundary_info = {
                'pooling': pool_type,
                'positive_boundary_cases': [all_filenames[i] for i in pos_boundary],
                'negative_boundary_cases': [all_filenames[i] for i in neg_boundary],
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            boundary_file = os.path.join(output_dir, f"boundary_cases_{pool_type}.json")
            with open(boundary_file, 'w') as f:
                json.dump(boundary_info, f, indent=2)
            print(f"\nSaved boundary cases to {boundary_file}")
        
        # Plot visualizations
        output_path_tsne = os.path.join(output_dir, f"tsne_{pool_type}_pooling.png")
        output_path_pca = os.path.join(output_dir, f"pca_{pool_type}_pooling.png")
        output_path_umap = os.path.join(output_dir, f"umap_{pool_type}_pooling.png")
        output_path_dist = os.path.join(output_dir, f"distance_dist_{pool_type}_pooling.png")
        
        # Generate and save plots
        tsne_results = plot_tsne(all_embeddings, all_labels, 
                               f"t-SNE of {pool_type.capitalize()} Pooled Embeddings\n(Blue=Negative, Yellow=Positive)",
                               output_path_tsne)
        
        pca_results = plot_pca(all_embeddings, all_labels,
                            f"PCA of {pool_type.capitalize()} Pooled Embeddings\n(Blue=Negative, Yellow=Positive)",
                            output_path_pca)
        
        umap_results = plot_umap(all_embeddings, all_labels,
                              f"UMAP of {pool_type.capitalize()} Pooled Embeddings\n(Blue=Negative, Yellow=Positive)",
                              output_path_umap)
        
        plot_distance_distributions(all_embeddings, all_labels,
                                 f"{pool_type.capitalize()} Pooling",
                                 output_path_dist)
        
        # Evaluate clustering
        metrics = evaluate_clustering(all_embeddings, all_labels, pool_type)
        all_metrics.append(metrics)
    
    # Print clustering metrics
    print("\nClustering Metrics:")
    print("-" * 80)
    for metric in all_metrics:
        print(f"\n{metric['pooling_type'].upper()} Pooling:")
        print("  K-means:")
        print(f"    Silhouette Score: {metric['kmeans_silhouette']:.4f}")
        print(f"    Calinski-Harabasz Index: {metric['kmeans_calinski']:.4f}")
        print(f"    Davies-Bouldin Index: {metric['kmeans_davies']:.4f}")
        print("  DBSCAN:")
        print(f"    Silhouette Score: {metric['dbscan_silhouette'] if metric['dbscan_silhouette'] is not None else 'N/A'}")
        print(f"    Calinski-Harabasz Index: {metric['dbscan_calinski'] if metric['dbscan_calinski'] is not None else 'N/A'}")
        print(f"    Davies-Bouldin Index: {metric['dbscan_davies'] if metric['dbscan_davies'] is not None else 'N/A'}")
    
    # Save metrics to file after converting numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj

    metrics_path = os.path.join(output_dir, "clustering_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(convert_numpy(all_metrics), f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    print("\nAnalysis complete! Check the 'pooling_comparison' directory for results.")

if __name__ == "__main__":
    main()
