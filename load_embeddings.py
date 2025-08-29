import numpy as np
import os
from pathlib import Path

def load_embeddings(folder_path):
    """Load all .npy embedding files from the specified folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    embeddings = {}
    for file_path in folder.glob('*.npy'):
        try:
            # Load the embedding
            embedding = np.load(file_path)
            # Store with filename (without extension) as key
            embeddings[file_path.stem] = embedding
            print(f"Loaded {file_path.name}: shape={embedding.shape}, dtype={embedding.dtype}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return embeddings

def analyze_embeddings(embeddings_dict):
    """Print analysis of the loaded embeddings."""
    if not embeddings_dict:
        print("No embeddings found to analyze.")
        return
    
    print("\n=== Embedding Analysis ===")
    print(f"Number of embeddings: {len(embeddings_dict)}")
    
    # Get first embedding to show details
    first_key = next(iter(embeddings_dict))
    first_embedding = embeddings_dict[first_key]
    
    print(f"\nFirst embedding ({first_key}):")
    print(f"  Shape: {first_embedding.shape}")
    print(f"  Data type: {first_embedding.dtype}")
    print(f"  Min value: {np.min(first_embedding):.4f}")
    print(f"  Max value: {np.max(first_embedding):.4f}")
    print(f"  Mean: {np.mean(first_embedding):.4f}")
    print(f"  Std dev: {np.std(first_embedding):.4f}")

if __name__ == "__main__":
    # Path to your embeddings folder
    embeddings_folder = "public_dataset/embeddings"
    
    print(f"Loading embeddings from: {embeddings_folder}")
    try:
        # Load all embeddings
        all_embeddings = load_embeddings(embeddings_folder)
        
        if all_embeddings:
            # Analyze the loaded embeddings
            analyze_embeddings(all_embeddings)
            
            # Example: Access the first embedding
            first_key = next(iter(all_embeddings))
            print(f"\nFirst few values of {first_key}:")
            print(all_embeddings[first_key][:5])  # Print first 5 values
            
            # Save all embeddings to a single file if needed
            # output_path = "all_embeddings.npz"
            # np.savez_compressed(output_path, **all_embeddings)
            # print(f"\nSaved all embeddings to {output_path}")
        else:
            print("No .npy files found in the specified folder.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
