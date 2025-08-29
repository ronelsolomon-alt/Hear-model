import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import torch
import torchaudio
from pathlib import Path
import glob

# Configuration
SAMPLE_RATE = 16000  # Target sample rate for audio
WINDOW_SIZE = 2.0    # Window size in seconds
HOP_SIZE = 1.0       # Hop size in seconds for sliding window
OUTPUT_DIR = "sliding_window_embeddings"

def load_audio(audio_path, target_sr=SAMPLE_RATE):
    """Load and resample audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {audio_path}: {str(e)}")
        return None, None

def create_sliding_windows(audio, sr, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    """Create sliding windows from audio signal.
    
    Args:
        audio: Numpy array of audio samples
        sr: Sample rate in Hz
        window_size: Window size in seconds
        hop_size: Hop size in seconds
        
    Returns:
        List of window dictionaries with audio and metadata
    """
    import numpy as np
    
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    audio_duration = len(audio) / sr
    
    print(f"Creating windows - Duration: {audio_duration:.2f}s, "
          f"Window: {window_size}s, Hop: {hop_size}s, "
          f"Samples: {len(audio)}")
    
    # Special handling for 1-second audio when window_size is 2 seconds
    if abs(audio_duration - 1.0) < 0.1 and abs(window_size - 2.0) < 0.1:
        print("1-second audio detected - padding with silence to 2 seconds")
        # Pad with silence to reach 2 seconds
        padding = np.zeros(window_samples - len(audio))
        padded_audio = np.concatenate([audio, padding])
        return [{
            'audio': padded_audio,
            'start_time': 0.0,
            'end_time': window_size,
            'window_idx': 0,
            'is_full_window': True,
            'was_padded': True
        }]
    
    # Handle case where audio is exactly or very close to window size
    duration_diff = abs(audio_duration - window_size)
    if duration_diff < 0.1:  # Within 100ms of window size
        print(f"Audio is exactly {audio_duration:.2f}s - creating single window")
        return [{
            'audio': audio,
            'start_time': 0.0,
            'end_time': audio_duration,
            'window_idx': 0,
            'is_full_window': True,
            'was_padded': False
        }]
    
    # Handle case where audio is shorter than window size
    if len(audio) < window_samples:
        print(f"Audio too short: {audio_duration:.2f}s < {window_size}s")
        # Pad with silence to match window size
        padding = np.zeros(window_samples - len(audio))
        padded_audio = np.concatenate([audio, padding])
        return [{
            'audio': padded_audio,
            'start_time': 0.0,
            'end_time': window_size,
            'window_idx': 0,
            'is_full_window': False,
            'was_padded': True
        }]
    
    # Calculate number of complete windows
    n_frames = 1 + int((len(audio) - window_samples) / hop_samples)
    windows = []
    
    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        
        # Skip incomplete windows at the end
        if end > len(audio):
            continue
            
        window = audio[start:end]
        windows.append({
            'audio': window,
            'start_time': start / sr,
            'end_time': end / sr,
            'window_idx': i,
            'is_full_window': True
        })
    
    # If no windows were created but we have enough samples, create one window
    if not windows and len(audio) >= window_samples:
        print("No complete windows created, using first window")
        return [{
            'audio': audio[:window_samples],
            'start_time': 0.0,
            'end_time': window_size,
            'window_idx': 0,
            'is_full_window': True
        }]
    
    print(f"Created {len(windows)} windows")
    return windows

def extract_embeddings(audio_windows, model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Extract embeddings for each window using the HEAR model.
    
    Raises:
        Exception: If there's an error processing any window
    """
    from new import get_hear_embeddings
    
    embeddings = []
    
    for window in tqdm(audio_windows, desc="Extracting HEAR embeddings"):
        try:
            # Get HEAR embeddings - pass the audio data directly
            embedding = get_hear_embeddings(window['audio'])
            
            if embedding is not None:
                # Ensure embedding is a numpy array with the right shape
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                
                # Flatten the embedding to 1D array if it's 2D
                if len(embedding.shape) > 1 and embedding.shape[0] == 1:
                    embedding = embedding.flatten()
                
                # Ensure the embedding has the expected size (512)
                if len(embedding) < 512:
                    # Pad with zeros if too small
                    padding = np.zeros(512 - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                elif len(embedding) > 512:
                    # Truncate if too large
                    embedding = embedding[:512]
                
                embeddings.append({
                    'embedding': embedding,
                    'start_time': window['start_time'],
                    'end_time': window['end_time'],
                    'window_idx': window['window_idx']
                })
            else:
                # If no embedding was returned, add a zero vector as placeholder
                print(f"Warning: No embedding returned for window {window['window_idx']}")
                embeddings.append({
                    'embedding': np.zeros(512),  # Standard size for HEAR embeddings
                    'start_time': window['start_time'],
                    'end_time': window['end_time'],
                    'window_idx': window['window_idx'],
                    'error': 'No embedding returned'
                })
                error_msg = f"No embedding returned for window {window['window_idx']}"
                print(f"\nError: {error_msg}")
                raise Exception(error_msg)
            
        except Exception as e:
            print(f"\nError processing window {window['window_idx']}: {str(e)}")
            # Add a zero vector as placeholder if there's an error
            embeddings.append({
                'embedding': np.zeros(512),  # Standard size for HEAR embeddings
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'window_idx': window['window_idx'],
                'error': str(e)
            })
            error_msg = f"Error processing window {window['window_idx']}: {str(e)}"
            print(f"\n{error_msg}")
            raise Exception(error_msg) from e
    
    return embeddings

def save_embeddings(embeddings, audio_path, output_dir):
    """Save embeddings and metadata to disk."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Save embeddings
    try:
        embeddings_array = np.array([e['embedding'] for e in embeddings])
        np.save(os.path.join(output_dir, f"{base_name}_embeddings.npy"), embeddings_array)
    except Exception as e:
        print(f"Error saving embeddings for {base_name}: {str(e)}")
        return None
    
    # Save metadata
    metadata = []
    for e in embeddings:
        try:
            # Safely get the embedding shape, default to empty tuple if not available
            if 'embedding' in e and hasattr(e['embedding'], 'shape'):
                embedding_shape = list(e['embedding'].shape)
            else:
                embedding_shape = []
            
            metadata.append({
                'window_idx': e.get('window_idx', -1),
                'start_time': e.get('start_time', 0.0),
                'end_time': e.get('end_time', 0.0),
                'embedding_shape': embedding_shape,
                'error': e.get('error', None)  # Include any error messages
            })
        except Exception as e_meta:
            print(f"Error creating metadata for window {e.get('window_idx', 'unknown')}: {str(e_meta)}")
            continue
    
    try:
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e_meta_save:
        print(f"Error saving metadata for {base_name}: {str(e_meta_save)}")
        return None
    
    return os.path.join(output_dir, f"{base_name}_embeddings.npy")

def process_audio_file(audio_info, output_dir, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    """Process a single audio file and save its windowed embeddings.
    
    Args:
        audio_info: Tuple of (audio_path, class_label)
        output_dir: Output directory for embeddings
        window_size: Size of sliding window in seconds (default: 2.0)
        hop_size: Hop size between windows in seconds (default: 1.0)
        
    Returns:
        str: Path to the saved embeddings file, or None if processing failed
        
    Raises:
        Exception: If there's an error processing the audio file
    """
    audio_path, class_label = audio_info
    print(f"\nProcessing: {audio_path}")
    
    try:
        # Load audio with explicit sample rate
        audio, sr = load_audio(audio_path)
        if audio is None:
            raise Exception(f"Failed to load audio file: {audio_path}")
        
        # Print debug info
        audio_duration = len(audio) / sr
        print(f"Audio duration: {audio_duration:.2f}s, Sample rate: {sr}Hz, Samples: {len(audio)}")
        
        # Create sliding windows
        windows = create_sliding_windows(audio, sr, window_size, hop_size)
        if not windows:
            # If no windows created, check if we can create at least one window
            min_samples = int(window_size * sr)
            if len(audio) >= min_samples:
                # Create a single window from the first min_samples
                windows = [{
                    'audio': audio[:min_samples],
                    'start_time': 0.0,
                    'end_time': window_size,
                    'window_idx': 0
                }]
                print(f"Created single window from first {window_size}s")
            else:
                raise Exception(f"Audio too short: {audio_duration:.2f}s < {window_size}s")
        
        print(f"Created {len(windows)} windows")
        
        # Extract embeddings for each window
        embeddings = extract_embeddings(windows, model=None)
        if not embeddings:
            raise Exception("No embeddings generated")
        
        # Add class information to embeddings
        for emb in embeddings:
            emb['class_label'] = class_label
        
        # Create class-specific subdirectories
        class_dir = os.path.join(output_dir, 'positive' if class_label == 1 else 'negative')
        os.makedirs(class_dir, exist_ok=True)
        
        # Save embeddings
        output_path = save_embeddings(embeddings, audio_path, class_dir)
        if output_path is None:
            raise Exception("Failed to save embeddings")
        
        print(f"Successfully processed and saved embeddings to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate sliding window embeddings for audio files')
    parser.add_argument('--audio-dir', type=str, required=True, 
                       help='Directory containing audio files (should have Positive/ and Negative/ subdirectories)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory for embeddings (default: {OUTPUT_DIR})')
    parser.add_argument('--window-size', type=float, default=WINDOW_SIZE,
                       help=f'Window size in seconds (default: {WINDOW_SIZE})')
    parser.add_argument('--hop-size', type=float, default=HOP_SIZE,
                       help=f'Hop size between windows in seconds (default: {HOP_SIZE})')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop processing on first error (default: continue on error)')
    parser.add_argument('--max-files', type=int, default=0,
                       help='Maximum number of files to process per class (0 for all)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*50)
    print(f"Audio Directory: {args.audio_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Window Size: {args.window_size}s")
    print(f"Hop Size: {args.hop_size}s")
    print(f"Stop on Error: {args.stop_on_error}")
    print("="*50 + "\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'negative'), exist_ok=True)
    
    # Get list of audio files with their class labels
    audio_files = []
    for class_dir, class_label in [('Positive', 1), ('Negative', 0)]:
        class_path = os.path.join(args.audio_dir, class_dir)
        if not os.path.exists(class_path):
            print(f"Warning: Directory not found: {class_path}")
            continue
            
        # Find all audio files in the class directory
        class_files = []
        for ext in ['*.wav', '*.mp3']:
            class_files.extend(glob.glob(os.path.join(class_path, '**', ext), recursive=True))
        
        # Sort for reproducibility and limit number of files if specified
        class_files = sorted([f for f in class_files if os.path.isfile(f)])
        if args.max_files > 0:
            class_files = class_files[:args.max_files]
            
        print(f"Found {len(class_files)} {class_dir} files")
        audio_files.extend([(f, class_label) for f in class_files])
    
    if not audio_files:
        print("Error: No audio files found!")
        return
        
    print(f"\nTotal files to process: {len(audio_files)}")
    
    # Process each audio file with progress tracking
    success_count = 0
    for i, (audio_path, class_label) in enumerate(audio_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing file {i}/{len(audio_files)}: {os.path.basename(audio_path)}")
        print(f"Class: {'Positive' if class_label == 1 else 'Negative'}")
        
        try:
            output_path = process_audio_file(
                (audio_path, class_label),
                args.output_dir,
                window_size=args.window_size,
                hop_size=args.hop_size
            )
            if output_path:
                success_count += 1
                print(f"✅ Successfully processed {os.path.basename(audio_path)}")
            else:
                print(f"❌ Failed to process {os.path.basename(audio_path)}")
                if args.stop_on_error:
                    print("Stopping due to error")
                    break
                    
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            break
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(audio_path)}: {str(e)}")
            if args.stop_on_error:
                print("Stopping due to error")
                break
    
    # Print summary
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(audio_files)} files")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("="*50)

if __name__ == "__main__":
    main()
    # python generate_sliding_window_embeddings.py \--audio-dir /Users/ronel/Downloads/dev/templates/hearpython/cleaned_data \--output-dir sliding_window_embeddings
