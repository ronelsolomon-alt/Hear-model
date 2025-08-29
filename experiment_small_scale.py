import numpy as np
from pathlib import Path
import joblib
import librosa
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from train_improved_model import load_embeddings, train_improved_model
import random
import os
import soundfile as sf
import tempfile
import json
import datetime
import shutil
import time
from tqdm import tqdm
from new import get_hear_embeddings
from pydub import AudioSegment
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf  # For compute_spectrogram and compute_loudness functions
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    make_scorer, roc_curve, auc, precision_recall_curve, average_precision_score,
    cohen_kappa_score, mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, mean_squared_log_error, median_absolute_error, 
    mean_absolute_percentage_error, mean_squared_log_error, max_error, 
    mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from scipy import stats
from scipy.stats import ttest_ind
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism to learn the importance of different segments.
    """
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
    def forward(self, x):
        # x shape: (seq_len, input_dim)
        attention_weights = self.attention(x)  # (seq_len, 1)
        weighted_embeddings = x * attention_weights  # (seq_len, input_dim)
        return weighted_embeddings.sum(dim=0), attention_weights.squeeze()

def extract_aggregated_features(embeddings, use_attention=True):
    """
    Extract aggregated features from embeddings using attention-based pooling.
    
    Args:
        embeddings: Numpy array of shape (n_windows, embedding_dim)
        use_attention: Whether to use attention-based pooling (True) or mean+std (False)
        
    Returns:
        Numpy array of aggregated features
    """
    if len(embeddings) == 0:
        return None
    
    if use_attention:
        # Convert to PyTorch tensor
        embeddings_tensor = torch.FloatTensor(embeddings)
        
        # Initialize attention model
        attention_model = AttentionPooling(input_dim=embeddings.shape[1])
        
        # Get attention-weighted features
        with torch.no_grad():
            weighted_embedding, _ = attention_model(embeddings_tensor)
            aggregated = weighted_embedding.numpy()
    else:
        # Fallback to mean and std
        mean_emb = np.mean(embeddings, axis=0)
        std_emb = np.std(embeddings, axis=0)
        aggregated = np.concatenate([mean_emb, std_emb])
    
    return aggregated

def spectral_gate(y, sr, n_fft=2048, hop_length=512, threshold=0.1):
    """Apply spectral gating noise reduction."""
    # Compute spectrogram
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Calculate noise profile (first few frames assumed to be noise)
    noise_profile = np.median(D[:, :5], axis=1, keepdims=True)
    
    # Apply threshold
    mask = D > (noise_profile * threshold)
    y_denoised = librosa.istft(D * mask, hop_length=hop_length)
    
    return y_denoised

def safe_mean(x, default=0.0):
    """Safely compute mean with NaN and empty array handling."""
    if len(x) == 0 or np.isnan(x).all():
        return default
    return np.nanmean(x)

def safe_std(x, default=0.0):
    """Safely compute standard deviation with NaN and empty array handling."""
    if len(x) == 0 or np.isnan(x).all():
        return default
    return np.nanstd(x)

def safe_max(x, default=0.0):
    """Safely compute max with NaN and empty array handling."""
    if len(x) == 0 or np.isnan(x).all():
        return default
    return np.nanmax(x)

def extract_audio_features(y, sr, n_mfcc=20, n_mels=128):
    """
    Extract comprehensive audio features with enhanced discrimination.
    
    Args:
        y: Audio time series
        sr: Sample rate
        n_mfcc: Number of MFCCs to extract
        n_mels: Number of mel bands to generate
        
    Returns:
        np.ndarray: Extracted features as a 1D array
    """
    features = {}
    
    try:
        # Input validation
        if len(y) == 0:
            raise ValueError("Empty audio signal")
            
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=0.0)
            
        if np.all(y == 0):
            # Return array of zeros with expected length if silent audio
            return np.zeros(13 + n_mfcc * 2 + 20)  # Adjust size as needed
        
        # 1. Time-domain features with NaN handling
        try:
            features.update({
                'zcr': safe_mean(librosa.feature.zero_crossing_rate(y)),
                'rms': np.sqrt(safe_mean(y**2)),
                'amplitude_envelope': safe_max(np.abs(y)),
                'energy': safe_mean(y**2),
                'autocorrelation': safe_max(np.correlate(y, y, mode='same'))
            })
        except Exception as e:
            print(f"Error in time-domain features: {str(e)}")
            features.update({k: 0.0 for k in ['zcr', 'rms', 'amplitude_envelope', 'energy', 'autocorrelation']})
        
        # 2. Frequency-domain features with error handling
        try:
            S = np.abs(librosa.stft(y))
            S_dB = librosa.amplitude_to_db(S, ref=np.max)
            
            # Spectral features with NaN handling
            spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
            
            features.update({
                'spectral_centroid_mean': safe_mean(spectral_centroid),
                'spectral_bandwidth_mean': safe_mean(spectral_bandwidth),
                'spectral_rolloff_mean': safe_mean(spectral_rolloff),
                'spectral_flatness': safe_mean(librosa.feature.spectral_flatness(S=S)),
                'spectral_contrast': safe_mean(librosa.feature.spectral_contrast(S=S, sr=sr)),
            })
        except Exception as e:
            print(f"Error in frequency-domain features: {str(e)}")
            features.update({k: 0.0 for k in ['spectral_centroid_mean', 'spectral_bandwidth_mean', 
                                            'spectral_rolloff_mean', 'spectral_flatness', 'spectral_contrast']})
        
        # 3. MFCC features with error handling
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            for i, mfcc in enumerate(mfccs):
                features[f'mfcc_{i}_mean'] = safe_mean(mfcc)
                features[f'mfcc_{i}_std'] = safe_std(mfcc)
        except Exception as e:
            print(f"Error in MFCC features: {str(e)}")
            for i in range(n_mfcc):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
        
        # 4. Temporal features with error handling
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features.update({
                'onset_strength_mean': safe_mean(onset_env),
                'onset_strength_std': safe_std(onset_env),
                'onset_rate': safe_mean(librosa.onset.onset_detect(y=y, sr=sr, units='time'))
            })
        except Exception as e:
            print(f"Error in temporal features: {str(e)}")
            features.update({'onset_strength_mean': 0.0, 'onset_strength_std': 0.0, 'onset_rate': 0.0})
        
        # 5. Harmonic-percussive separation with error handling
        try:
            y_harm, y_perc = librosa.effects.hpss(y)
            harm_energy = safe_mean(y_harm**2)
            perc_energy = safe_mean(y_perc**2)
            total_energy = harm_energy + perc_energy + 1e-10
            
            features.update({
                'harmonic_ratio': harm_energy / total_energy,
                'percussive_ratio': perc_energy / total_energy
            })
        except Exception as e:
            print(f"Error in harmonic-percussive separation: {str(e)}")
            features.update({'harmonic_ratio': 0.5, 'percussive_ratio': 0.5})
        
        # 6. Statistical features with error handling
        try:
            features.update({
                'kurtosis': safe_mean([stats.kurtosis(y) if len(y) > 3 else 0]),
                'skewness': safe_mean([stats.skew(y) if len(y) > 2 else 0]),
                'entropy': safe_mean([stats.entropy(np.histogram(y, bins=20)[0] + 1e-10)])
            })
        except Exception as e:
            print(f"Error in statistical features: {str(e)}")
            features.update({'kurtosis': 0.0, 'skewness': 0.0, 'entropy': 0.0})
        
        # Convert all features to float and handle any remaining NaNs/Infs
        features = {k: float(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)) 
                   for k, v in features.items()}
        
        return np.array(list(features.values()))
        
    except Exception as e:
        print(f"Critical error in extract_audio_features: {str(e)}")
        # Return array of zeros with expected length on critical failure
        return np.zeros(13 + n_mfcc * 2 + 20)  # Adjust size to match expected output

def compute_spectrogram(
    audio: np.ndarray | tf.Tensor,
    frame_length: int = 400,
    frame_step: int = 160,
):
    """Compute the spectrogram of an audio signal.
    
    Args:
        audio: Input audio signal as a numpy array or TensorFlow tensor.
        frame_length: Length of the FFT window in samples.
        frame_step: Number of samples between successive frames.
        
    Returns:
        A tensor containing the magnitude spectrogram.
    """
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    elif len(audio.shape) > 2:
        raise NotImplementedError(
            f'`audio` should have at most 2 dimensions but had {len(audio.shape)}')
    stft_output = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length)
    spectrogram = tf.abs(stft_output)
    return spectrogram


def compute_loudness(
    audio: np.ndarray | tf.Tensor,
    sample_rate: float = 16000.0,
):
    """Computes loudness.

    It is defined as the per-channel per-timestep cross-frequency L2 norm of the
    log mel spectrogram.

    Args:
        audio: Array of shape [num_timesteps] representing a raw wav file.
        sample_rate: The sample rate of the input audio.

    Returns:
        An array of shape [num_timesteps] representing the loudness in decibels.
    """
    frame_step = int(sample_rate) // 100  # 10 ms
    frame_length = 25 * int(sample_rate) // 1000  # 25 ms
    linear_spectrogram = compute_spectrogram(
        audio.astype(np.float32),
        frame_length=frame_length,
        frame_step=frame_step,
    )
    print(audio.shape, audio.shape[0] // 16000, linear_spectrogram.shape)
    sum_amplitude = np.sum(linear_spectrogram, axis=1)
    loudness_db_timeseries = 20 * np.log10(sum_amplitude)
    return np.asarray(loudness_db_timeseries)


def get_all_embeddings(audio_path, target_sr=16000, window_duration=2.0, hop_ratio=0.5):
    """
    Process an audio file using a sliding window approach to get embeddings for each window.
    
    Args:
        audio_path (str or Path): Path to the audio file
        target_sr (int): Target sample rate
        window_duration (float): Duration of each window in seconds
        hop_ratio (float): Hop size as a ratio of window duration (0-1)
    
    Returns:
        tuple: (embeddings, timestamps, loudness_analysis) where:
            - embeddings: List of embeddings for each window
            - timestamps: List of timestamps for each window
            - loudness_analysis: Dictionary containing loudness analysis results
    """
    try:
        # Convert audio_path to Path object
        audio_path = Path(audio_path)
        
        # Create output directory for loudness plots
        output_dir = Path('loudness_analysis')
        output_dir.mkdir(exist_ok=True)
        
        # Load audio with librosa and normalize amplitude
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Analyze loudness for the current audio file
        analyze_audio_loudness(audio_path, output_dir=output_dir)
        
        # Apply noise reduction
        y_denoised = spectral_gate(y, sr)
        
        # Normalize amplitude to range [-1, 1]
        y = librosa.util.normalize(y_denoised)
        
        # Analyze loudness for the entire audio file
        loudness = compute_loudness(y, sample_rate=sr)
        
        # Create a plot of the loudness
        plt.figure(figsize=(12, 6))
        plt.plot(loudness, label='Loudness (dB)')
        plt.axhline(42, c='r', linestyle='--', label='Threshold (42 dB)')
        plt.title(f'Loudness Analysis - {audio_path.name}')
        plt.xlabel('Time (frames)')
        plt.ylabel('Loudness (dB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = output_dir / f'loudness_{audio_path.stem}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Prepare loudness analysis results
        loudness_analysis = {
            'mean_loudness': float(np.mean(loudness)),
            'max_loudness': float(np.max(loudness)),
            'min_loudness': float(np.min(loudness)),
            'loudness_std': float(np.std(loudness)),
            'plot_path': str(plot_path.absolute())
        }
        
        # Calculate samples per window and hop
        samples_per_window = int(window_duration * sr)
        hop_size = int(samples_per_window * hop_ratio)
        
        # Ensure we have at least one window
        if len(y) < samples_per_window:
            # Pad if audio is shorter than window
            padding = np.zeros(samples_per_window - len(y))
            y = np.concatenate([y, padding])
            
        
        # Process with fixed 2-second windows as required by HEAR endpoint
        embeddings = []
        timestamps = []
        win_duration = 2.0  # Fixed 2-second window
        win_samples = int(win_duration * sr)
        hop_size = int(win_samples * hop_ratio)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(0, max(1, len(y) - win_samples + 1), hop_size):
                # Extract window
                window = y[i:i + win_samples]
                
                # Ensure exactly 2 seconds of audio (32,000 samples at 16kHz)
                if len(window) < win_samples:
                    # Pad with zeros if too short
                    padding = np.zeros(win_samples - len(window))
                    window = np.concatenate([window, padding])
                elif len(window) > win_samples:
                    # Truncate if too long
                    window = window[:win_samples]
                
                # Apply window function
                window = window * np.hanning(len(window))
                
                # Save and process the window
                chunk_path = os.path.join(temp_dir, f'window_{i}.wav')
                sf.write(chunk_path, window, sr)
                
                # Get HEAR embeddings
                try:
                    chunk_embedding = get_hear_embeddings(chunk_path)
                    
                    if chunk_embedding is not None:
                        # Extract additional audio features
                        audio_features = extract_audio_features(window, sr)
                        
                        # Combine HEAR embeddings with custom features
                        combined_features = np.concatenate([
                            chunk_embedding[0],  # HEAR embeddings
                            audio_features      # Custom audio features
                        ])
                        
                        embeddings.append(combined_features)
                        timestamps.append(i/sr)  # Store start time of each window
                        print(f"Processed 2.0s window starting at {i/sr:.1f}s")
                    else:
                        print(f"Warning: Failed to get embeddings for window starting at {i/sr:.1f}s")
                except Exception as e:
                    print(f"Error processing window starting at {i/sr:.1f}s: {str(e)}")
        
        if not embeddings:
            print("No valid cough segments found")
            return None, None, None
            
        # Check if all embeddings have the same shape
        first_shape = np.array(embeddings[0]).shape
        for i, emb in enumerate(embeddings[1:], 1):
            if np.array(emb).shape != first_shape:
                print(f"Warning: Inconsistent embedding shapes at index {i}. Expected {first_shape}, got {np.array(emb).shape}")
                return None, None, None
        
        # Convert to numpy arrays if shapes are consistent
        embeddings_array = np.array(embeddings) if embeddings else np.array([])
        timestamps_array = np.array(timestamps) if timestamps else np.array([])
        
        return embeddings_array, timestamps_array, loudness_analysis
        
    except Exception as e:
        print(f"Error in get_all_embeddings for {audio_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def process_and_save_embeddings(audio_path, output_dir, class_label=None, max_files=None, use_attention=True):
    """
    Process audio files in a directory, save their embeddings (both raw and processed),
    and copy the original audio files. Skips files that have already been processed.
    
    Args:
        audio_path (str): Path to directory containing audio files
        output_dir (str): Base directory to save outputs (will create 'audio', 'embeddings', and 'raw_embeddings' subdirectories)
        class_label (int, optional): Label for the class (0 or 1). Defaults to None.
        max_files (int, optional): Maximum number of files to process
        use_attention (bool, optional): Whether to use attention-based aggregation. Defaults to True.
    
    Returns:
        tuple: (X, y) where X is the array of processed embeddings and y is the labels
    """
    # Convert output_dir to Path object and create subdirectories
    output_dir = Path(output_dir)
    embeddings_dir = output_dir / 'embeddings'
    raw_embeddings_dir = output_dir / 'raw_embeddings'
    
    # Create directories if they don't exist
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    raw_embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have 100 files in both directories
    if len(list(embeddings_dir.glob('*.npy'))) >= 100 and len(list(raw_embeddings_dir.glob('*.npz'))) >= 100:
        print(f"Found 100+ files in both {embeddings_dir} and {raw_embeddings_dir}. Skipping embedding generation.")
        # Load existing embeddings
        X = []
        y = []
        for emb_file in embeddings_dir.glob('*.npy'):
            try:
                emb_data = np.load(emb_file, allow_pickle=True).item()
                if isinstance(emb_data, dict) and 'embedding' in emb_data:
                    X.append(emb_data['embedding'])
                    y.append(emb_data.get('label', class_label if class_label is not None else -1))
            except Exception as e:
                print(f"Error loading {emb_file}: {e}")
        
        if len(X) > 0:
            return np.array(X), np.array(y)
        return np.array([]), np.array([])
    
    # Check if librosa is available
    try:
        import librosa
        HAS_LIBROSA = True
    except ImportError:
        print("Warning: librosa not available. Some audio processing features will be disabled.")
        HAS_LIBROSA = False
    # Initialize variables
    file_id = None
    class_label = class_label if class_label is not None else -1  # Default value when label is missing
    metadata = []  # Initialize metadata list to store metadata for each processed file
    X = []  # List to store feature vectors
    y = []  # List to store corresponding labels
    processed_count = 0  # Counter for successfully processed files
    skipped_count = 0  # Counter for skipped files
    
    # Create audio output directory if it doesn't exist
    audio_output_dir = output_dir / 'audio'
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of audio files
    audio_files = list(Path(audio_path).glob('*.wav')) + list(Path(audio_path).glob('*.mp3'))
    if max_files and len(audio_files) > max_files:
        audio_files = random.sample(audio_files, max_files)
    
    all_embeddings = []
    processed_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Skip if already processed
            output_base = os.path.splitext(audio_file.name)[0]
            output_embedding = Path(embeddings_dir) / f"{output_base}.npy"
            output_raw_embedding = Path(raw_embeddings_dir) / f"{output_base}_raw.npz"
            
            # Ensure output directories exist
            output_embedding.parent.mkdir(parents=True, exist_ok=True)
            output_raw_embedding.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if both processed and raw embeddings exist
            if os.path.exists(output_embedding) and os.path.exists(output_raw_embedding):
                # Load existing processed embedding
                embeddings = np.load(output_embedding, allow_pickle=True).item()
                if isinstance(embeddings, dict) and 'embedding' in embeddings:
                    all_embeddings.append(embeddings['embedding'])
                    processed_files.append(audio_file)
                continue
            
            try:
                # Process the audio file to get embeddings and timestamps
                result = get_all_embeddings(audio_file)
                if result is None or len(result) != 3:
                    print(f"Skipping {audio_file} - no embeddings generated or invalid return format")
                    continue
                
                # Unpack the result
                embeddings, timestamps, loudness_analysis = result
                
                # Validate the results
                if not isinstance(embeddings, (list, np.ndarray)) or len(embeddings) == 0:
                    print(f"Skipping {audio_file} - no valid embeddings")
                    continue
                
                if not isinstance(timestamps, (list, np.ndarray)) or len(timestamps) != len(embeddings):
                    print(f"Warning: Mismatch between number of embeddings ({len(embeddings)}) and timestamps ({len(timestamps) if timestamps is not None else 0})")
                    # Generate default timestamps if needed
                    if not timestamps:
                        timestamps = [i * 2.0 for i in range(len(embeddings))]  # 2-second windows
                
                if loudness_analysis is None:
                    loudness_analysis = {}
                    
                # Verify embeddings are valid
                if not isinstance(embeddings, (list, np.ndarray)) or len(embeddings) == 0:
                    print(f"No valid embeddings generated for {audio_file}")
                    continue
                
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                print(traceback.format_exc())
                continue
            
            if len(embeddings) == 0:
                print(f"Skipping {audio_file} - no valid segments found")
                continue
            
            # Ensure required fields exist
            if file_id is None:
                file_id = audio_file.stem  # Use filename as fallback ID
                print(f"Warning: file_id not provided, using filename as ID: {file_id}")
                
            if class_label is None:
                print(f"Warning: class_label not provided for file {file_id}, defaulting to -1")
                class_label = -1  # Default value when label is missing
            
            # Save raw embeddings with all chunk data
            raw_embeddings = {
                'embeddings': [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings],
                'timestamps': timestamps,
                'loudness': loudness_analysis,
                'file_path': str(audio_file.absolute()),
                'file_id': str(file_id),  # Ensure file_id is string
                'label': int(class_label),  # Ensure label is integer
                'timestamp': datetime.datetime.now().isoformat(),
                'sample_rate': 16000,  # Add sample rate for reference
                'window_duration': 2.0,  # Add window duration for reference
                'hop_ratio': 0.5  # Add hop ratio for reference
            }
            # Ensure output_raw_embedding is a Path object
            output_raw_embedding = Path(output_raw_embedding)
            output_raw_embedding.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(output_raw_embedding), **raw_embeddings)
            
            # Save processed embeddings (aggregated features)
            aggregated_embedding = extract_aggregated_features(embeddings, use_attention=True)
            if aggregated_embedding is None or np.isnan(aggregated_embedding).any():
                print(f"Skipping {audio_file} - invalid aggregated embedding")
                continue
                
            # Create output embedding path
            output_embedding = embeddings_dir / f"{audio_file.stem}.npz"
            output_embedding.parent.mkdir(parents=True, exist_ok=True)
            
            processed_embeddings = {
                'embedding': aggregated_embedding.tolist(),
                'file_path': str(audio_file.absolute()),
                'file_id': file_id,
                'label': class_label,
                'timestamp': datetime.datetime.now().isoformat(),
                'raw_embedding_path': str(output_raw_embedding.absolute()),
                'aggregation_method': 'attention' if use_attention else 'mean_std',
                'embedding_dim': len(aggregated_embedding)
            }
            np.savez_compressed(str(output_embedding), **processed_embeddings)
            
            # Copy audio file using consistent path handling
            audio_dest = audio_output_dir / audio_file.name
            if not audio_dest.exists():
                shutil.copy2(str(audio_file), str(audio_dest))
            
            # Get audio duration if librosa is available
            duration = None
            has_librosa = False
            try:
                import librosa
                has_librosa = True
            except ImportError:
                print("Warning: librosa not available. Some audio processing features will be disabled.")
            if has_librosa:
                try:
                    duration = librosa.get_duration(filename=str(audio_file))
                except Exception as e:
                    print(f"Warning: Could not get duration for {audio_file}: {e}")
            
            # Create metadata entry for this file
            metadata_entry = {
                'file_name': audio_file.name,
                'original_path': str(audio_file.absolute()),
                'embedding_path': str(output_embedding.absolute()),
                'raw_embedding_path': str(output_raw_embedding.absolute()),
                'label': class_label,
                'file_id': file_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'num_embeddings': len(embeddings),
                'sample_rate': 16000,
                'duration': duration,
                'channels': 1,  # Default to mono
                'file_size_bytes': audio_file.stat().st_size,
                'loudness_stats': {
                    'mean': float(np.mean(loudness_analysis['loudness'])),
                    'std': float(np.std(loudness_analysis['loudness'])),
                    'max': float(np.max(loudness_analysis['loudness'])),
                    'min': float(np.min(loudness_analysis['loudness']))
                } if loudness_analysis and 'loudness' in loudness_analysis else None,
                'processing_info': {
                    'window_duration': 2.0,
                    'hop_ratio': 0.5,
                    'aggregation_method': 'attention' if use_attention else 'mean_std',
                    'model': 'HEAR'
                }
            }
            metadata.append(metadata_entry)
            
            # Add to training data (use aggregated features: mean + std)
            agg_features = extract_aggregated_features(embeddings)
            if agg_features is not None:
                X.append(agg_features)
            y.append(class_label)
            
            # Periodically save metadata in case of failure
            # Save metadata at the end of processing
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    if metadata:
        save_metadata(metadata, output_dir / 'metadata.json')
       
    total_files = len(audio_files)
    failed_count = total_files - processed_count - skipped_count
    print(f"\nProcessing complete:")
    print(f"- Total files: {total_files}")
    print(f"- Successfully processed: {processed_count}")
    print(f"- Loaded existing: {skipped_count}")
    print(f"- Failed: {failed_count}")
    print(f"- Total embeddings: {len(X)}")
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    save_metadata(metadata, metadata_path)
    
    if not X:
        print("Warning: No embeddings were loaded or generated!")
        return np.array([]), np.array([])
    
    # Find the maximum length among all embeddings
    max_length = max(x.shape[0] for x in X if x is not None)
    
    # Pad or truncate embeddings to have consistent shape
    processed_X = []
    for x in X:
        if x is None:
            continue
            
        # If the embedding is shorter than max_length, pad with zeros
        if x.shape[0] < max_length:
            pad_width = [(0, max_length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1)
            x_padded = np.pad(x, pad_width, mode='constant')
            processed_X.append(x_padded)
        # If it's longer, truncate
        elif x.shape[0] > max_length:
            processed_X.append(x[:max_length])
        else:
            processed_X.append(x)
    
    # Convert to numpy arrays
    try:
        X_array = np.array(processed_X)
        y_array = np.array(y[:len(processed_X)])  # Ensure y has same length as X
        
        print(f"\nSuccessfully created arrays with shapes:")
        print(f"- X shape: {X_array.shape}")
        print(f"- y shape: {y_array.shape}")
        
        return X_array, y_array
        
    except Exception as e:
        print(f"Error creating numpy arrays: {str(e)}")
        print(f"Shapes of first 5 embeddings: {[x.shape for x in processed_X[:5]]}")
        return np.array([]), np.array([])

def save_metadata(metadata, path):
    """Save metadata to a JSON file with proper formatting."""
    try:
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")

def run_experiment():
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(f"experiment_results/experiment_{timestamp}")
    
    # Create directories for embeddings in the format expected by train_improved_model
    # Using relative paths within the project directory
    covid_embeddings_dir = Path('data/covid_positive/embeddings')
    non_covid_embeddings_dir = Path('data/non_covid_negative/embeddings')
    
    # Create parent directories if they don't exist
    covid_embeddings_dir.parent.mkdir(parents=True, exist_ok=True)
    non_covid_embeddings_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Process COVID samples (from cleaned_data/Positive)
    covid_path = Path('cleaned_data/Positive')
    print(f"\nProcessing COVID samples from: {covid_path.absolute()}")
    print(f"Directory exists: {covid_path.exists()}")
    if covid_path.exists():
        print(f"Files in directory: {list(covid_path.glob('*'))[:5]}...")
    
    X_covid, y_covid = process_and_save_embeddings(
        str(covid_path), str(covid_embeddings_dir.parent), class_label=1, max_files=100  # Process up to 100 files
    )
    print(f"COVID samples loaded: {len(X_covid) if X_covid is not None else 0}")
    
    # Process non-COVID samples (from cleaned_data/Negative)
    non_covid_path = Path('cleaned_data/Negative')
    print(f"\nProcessing non-COVID samples from: {non_covid_path.absolute()}")
    print(f"Directory exists: {non_covid_path.exists()}")
    if non_covid_path.exists():
        print(f"Files in directory: {list(non_covid_path.glob('*'))[:5]}...")
    
    X_non_covid, y_non_covid = process_and_save_embeddings(
        str(non_covid_path), str(non_covid_embeddings_dir.parent), class_label=0, max_files=100  # Process up to 100 files
    )
    print(f"Non-COVID samples loaded: {len(X_non_covid) if X_non_covid is not None else 0}")
    
    print(f"\nSuccessfully processed {len(X_covid)} COVID and {len(X_non_covid)} non-COVID samples")
    
    # Ensure both datasets have the same number of features
    min_features = min(X_covid.shape[1], X_non_covid.shape[1])
    
    # Truncate or pad features to match the minimum dimension
    if X_covid.shape[1] > min_features:
        X_covid = X_covid[:, :min_features]
    elif X_covid.shape[1] < min_features:
        pad_width = [(0, 0), (0, min_features - X_covid.shape[1])]
        X_covid = np.pad(X_covid, pad_width, mode='constant')
    
    if X_non_covid.shape[1] > min_features:
        X_non_covid = X_non_covid[:, :min_features]
    elif X_non_covid.shape[1] < min_features:
        pad_width = [(0, 0), (0, min_features - X_non_covid.shape[1])]
        X_non_covid = np.pad(X_non_covid, pad_width, mode='constant')
    
    # Combine the datasets
    X = np.vstack((X_covid, X_non_covid))
    y = np.concatenate((y_covid, y_non_covid))
    
    # Perform clustering analysis on the combined data
    print("\nPerforming clustering analysis on the combined dataset...")
    analyze_embeddings(X, y, output_dir=output_base)
    
    # Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Save the processed data for later use
    processed_data_dir = output_base / 'processed_data'
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(processed_data_dir / 'X_train.npy', X_train)
    np.save(processed_data_dir / 'X_test.npy', X_test)
    np.save(processed_data_dir / 'y_train.npy', y_train)
    np.save(processed_data_dir / 'y_test.npy', y_test)
    
    # Save experiment configuration
    experiment_info = {
        'experiment_timestamp': datetime.datetime.now().isoformat(),
        'num_covid_samples': len(X_covid),
        'num_non_covid_samples': len(X_non_covid),
        'train_test_split': '80/20',
        'input_directories': {
            'covid': str(Path('cleaned_data/Positive').absolute()),
            'non_covid': str(Path('cleaned_data/Negative').absolute())
        },
        'output_structure': {
            'base_dir': str(output_base.absolute()),
            'covid': {
                'audio': 'covid/audio/*',
                'embeddings': 'covid/embeddings/*.npy',
                'metadata': 'covid/metadata.json'
            },
            'non_covid': {
                'audio': 'non_covid/audio/*',
                'embeddings': 'non_covid/embeddings/*.npy',
                'metadata': 'non_covid/metadata.json'
            },
            'processed_data': 'processed_data/*.npy'
        },
        'model_info': {
            'embedding_dimension': X_train.shape[1] if len(X_train) > 0 else 0,
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test)
        }
    }
    
    with open(output_base / 'experiment_info.json', 'w') as f:
        json.dump(experiment_info, f, indent=2, default=str)
    
    # Prepare data for multiple classifiers
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    
    # Compare multiple classifiers
    print("\nComparing multiple classifiers...")
    results = compare_classifiers(X, y, test_size=0.2, random_state=42)
    
    # Save results to file
    results_df = pd.DataFrame(results).T
    results_file = output_base / 'classifier_comparison.csv'
    results_df.to_csv(results_file)
    print(f"\nSaved classifier comparison results to {results_file}")
    
    # Plot comparison
    plot_classifier_comparison(results_df, output_base)
    
    print("\nClassifier comparison complete. Results and visualizations saved to:")
    print(f"- {results_file}")
    print(f"- {output_base / 'classifier_comparison.png'}")

def analyze_embeddings(X, y, output_dir=None, n_clusters=2):
    """
    Analyze embeddings using clustering and similarity metrics.
    
    Args:
        X: Numpy array of shape (n_samples, n_features)
        y: Array-like of shape (n_samples,) with class labels
        output_dir: Directory to save results (if None, uses current directory)
        n_clusters: Number of clusters to use (default: 2 for binary classification)
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('.')
    
    results = {}
    
    # Encode labels if they're not numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 1. Clustering metrics
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Silhouette Score (higher is better)
        sil_score = silhouette_score(X, cluster_labels)
        results['silhouette_score'] = sil_score
        
        # Davies-Bouldin Index (lower is better)
        db_index = davies_bouldin_score(X, cluster_labels)
        results['davies_bouldin_index'] = db_index
        
        print("\n=== Clustering Metrics ===")
        print(f"Silhouette Score: {sil_score:.4f} (closer to 1 is better)")
        print(f"Davies-Bouldin Index: {db_index:.4f} (closer to 0 is better)")
        
        # Save clustering metrics to file
        metrics_file = output_dir / 'clustering_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'silhouette_score': float(sil_score),
                'davies_bouldin_index': float(db_index),
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_clusters': n_clusters,
                'class_distribution': dict(zip(*np.unique(y_encoded, return_counts=True)))
            }, f, indent=2)
            
        print(f"\nSaved clustering metrics to: {metrics_file}")
        
    except Exception as e:
        print(f"Error in clustering analysis: {str(e)}")
        results['clustering_error'] = str(e)
    
    # 2. Cosine Similarity Analysis
    try:
        # Calculate pairwise cosine similarities
        similarities = []
        within_class = []
        between_class = []
        
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                sim = 1 - cosine(X[i], X[j])
                similarities.append(sim)
                if y_encoded[i] == y_encoded[j]:
                    within_class.append(sim)
                else:
                    between_class.append(sim)
        
        # Calculate statistics
        results['mean_similarity'] = np.mean(similarities)
        results['mean_within_class_similarity'] = np.mean(within_class)
        results['mean_between_class_similarity'] = np.mean(between_class)
        
        # T-test for significance
        t_stat, p_value = ttest_ind(within_class, between_class, equal_var=False)
        results['similarity_ttest_pvalue'] = p_value
        
        print("\nCosine Similarity Analysis:")
        print(f"- Mean similarity: {results['mean_similarity']:.4f}")
        print(f"- Mean within-class similarity: {results['mean_within_class_similarity']:.4f}")
        print(f"- Mean between-class similarity: {results['mean_between_class_similarity']:.4f}")
        print(f"- T-test p-value: {p_value:.4f} (lower is better for class separation)")
        
        # Plot similarity distributions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(within_class, label='Within-class similarity', fill=True)
        sns.kdeplot(between_class, label='Between-class similarity', fill=True)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Distribution of Embedding Similarities')
        plt.legend()
        plt.savefig('similarity_distribution.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in similarity analysis: {str(e)}")
    
    return results

def compare_classifiers(X, y, test_size=0.2, random_state=42):
    """
    Compare multiple classifiers and return their performance metrics.
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    Returns:
        dict: Dictionary containing evaluation metrics for each classifier
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Define base classifiers
    classifiers = {
        "Support Vector Machine (linear)": SVC(kernel='linear', probability=True, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=128, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=128, random_state=random_state),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=random_state)
    }
    
    # Add Random Forest with Grid Search
    from sklearn.model_selection import GridSearchCV
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200, 128],
        'max_depth': [1, 2, 3, 5, 7, 6, 8, None],
        'min_samples_split': [2, 5, 10, None],
        'min_samples_leaf': [1, 2, 4, None],
        'max_features': ['sqrt', 'log2', 'None']
    }
    
    from sklearn.model_selection import StratifiedKFold
    
    # Create stratified k-fold object
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    gb = GradientBoostingClassifier(random_state=random_state)
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, 
                             cv=stratified_cv, n_jobs=-1, verbose=2, scoring='accuracy')
    
    # Add to classifiers with a special key
    classifiers["Random Forest (Grid Search)"] = grid_search
    
    results = {}
    
    # First fit the grid search to find best parameters
    print("\nPerforming Grid Search for Random Forest...")
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Get the best estimator and make predictions
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print("\nBest Model Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Update the best estimator in the classifiers dict
    classifiers["Random Forest (Best)"] = best_estimator
    
    for name, clf in classifiers.items():
        # Skip the grid search as we've already fit it
        if name == "Random Forest (Grid Search)":
            continue
            
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Create pipeline with scaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'train_time': train_time,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        
        results[name] = metrics
        
        # Print results
        print(f"\n{name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return results


def analyze_audio_loudness(audio_path, output_dir='loudness_analysis', loudness_threshold=42):
    """
    Analyze and plot loudness of a single audio file.
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save the plots
        loudness_threshold (float): Threshold for loudness in dB
        
    Returns:
        dict: Dictionary containing loudness statistics
    """
    from pathlib import Path
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute loudness
        loudness = compute_loudness(y, sr)
        
        # Calculate statistics
        stats = {
            'file': str(audio_path),
            'mean_loudness': float(np.mean(loudness)),
            'max_loudness': float(np.max(loudness)),
            'min_loudness': float(np.min(loudness)),
            'loudness_std': float(np.std(loudness))
        }
        
        # Generate plot
        plt.figure(figsize=(12, 6))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f'Waveform: {Path(audio_path).name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot loudness
        plt.subplot(2, 1, 2)
        times = np.linspace(0, len(y)/sr, len(loudness))
        plt.plot(times, loudness, label='Loudness (dB)')
        plt.axhline(y=loudness_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({loudness_threshold} dB)')
        plt.title('Loudness Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Loudness (dB)')
        plt.legend()
        
        # Save the plot
        plot_filename = output_dir / f"loudness_{Path(audio_path).stem}.png"
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Add plot path to stats
        stats['plot_path'] = str(plot_filename.absolute())
        
        return stats
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, None, None


def analyze_loudness(audio_dir, output_dir='loudness_analysis', is_test=False, max_plots=21, loudness_threshold=42):
    """
    Analyze and plot loudness of audio files.
    
    Args:
        audio_dir (str): Path to directory containing audio files (covid/non_covid)
        output_dir (str): Directory to save the plots
        is_test (bool): Whether this is test data
        max_plots (int): Maximum number of plots to generate per class
        loudness_threshold (float): Threshold for loudness in dB
    """
    from pathlib import Path
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create output directories
    base_dir = Path(output_dir) / ('test' if is_test else 'train')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing loudness for audio files in: {audio_dir}")
    
    # Get all WAV files from the directory
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise ValueError(f"Audio directory not found: {audio_dir}")
    
    # Find all WAV files in the directory and its subdirectories
    audio_files = list(audio_dir.rglob('*.wav'))
    if not audio_files:
        audio_files = list(audio_dir.rglob('*.ogg'))  # Try OGG if no WAV files found
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files for loudness analysis")
    
    # Calculate loudness for all files
    loudness_stats = []
    plot_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=None)
            
            # Compute loudness
            loudness = compute_loudness(y, sr)
            
            # Calculate statistics
            stats = {
                'file': str(audio_file),
                'mean_loudness': float(np.mean(loudness)),
                'max_loudness': float(np.max(loudness)),
                'min_loudness': float(np.min(loudness)),
                'loudness_std': float(np.std(loudness))
            }
            loudness_stats.append(stats)
            
            # Generate plot (up to max_plots)
            if plot_count < max_plots:
                plt.figure(figsize=(12, 6))
                
                # Plot waveform
                plt.subplot(2, 1, 1)
                librosa.display.waveshow(y, sr=sr, alpha=0.6)
                plt.title(f'Waveform: {audio_file.name}')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                
                # Plot loudness
                plt.subplot(2, 1, 2)
                times = np.linspace(0, len(y)/sr, len(loudness))
                plt.plot(times, loudness, label='Loudness (dB)')
                plt.axhline(y=loudness_threshold, color='r', linestyle='--', 
                           label=f'Threshold ({loudness_threshold} dB)')
                plt.title('Loudness Over Time')
                plt.xlabel('Time (s)')
                plt.ylabel('Loudness (dB)')
                plt.legend()
                
                # Save the plot
                plot_filename = base_dir / f"loudness_{audio_file.stem}.png"
                plt.tight_layout()
                plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
                plt.close()
                
                plot_count += 1
                
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Save loudness statistics to CSV
    if loudness_stats:
        import pandas as pd
        df = pd.DataFrame(loudness_stats)
        stats_file = base_dir / 'loudness_statistics.csv'
        df.to_csv(stats_file, index=False)
        print(f"\nSaved loudness statistics to: {stats_file}")
        
        # Print summary statistics
        print("\nLoudness Statistics Summary:")
        print(f"Average Loudness: {df['mean_loudness'].mean():.2f} dB")
        print(f"Maximum Loudness: {df['max_loudness'].max():.2f} dB")
        print(f"Minimum Loudness: {df['min_loudness'].min():.2f} dB")
        print(f"Files above {loudness_threshold} dB: {len(df[df['max_loudness'] > loudness_threshold])}/{len(df)} ({(len(df[df['max_loudness'] > loudness_threshold])/len(df)*100):.1f}%)")
    
    return loudness_stats


def plot_classifier_comparison(results_df, output_dir):
    """Create and save comparison plots for classifier performance"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            ax = axes[i]
            sns.barplot(
                x=results_df.index, 
                y=results_df[metric],
                ax=ax,
                palette='viridis'
            )
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylim(0, 1.1 if metric != 'train_time' else None)
            
            # Add value labels on top of bars
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width()/2.,
                    height + 0.02 if height < 0.1 else height * 1.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom'
                )
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = output_dir / 'classifier_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved classifier comparison plot to {plot_path}")

def compare_embeddings(positive_dir, negative_dir, output_dir='comparison_results'):
    """
    Compare embeddings between Positive and Negative classes.
    
    Args:
        positive_dir (str): Path to directory containing positive class embeddings
        negative_dir (str): Path to directory containing negative class embeddings
        output_dir (str): Directory to save comparison results
    """
    import os
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    def load_embeddings(embeddings_dir, max_attempts=5):
        """
        Load and aggregate all embeddings from a directory with NaN handling.
        If a file has NaN values, it will try up to max_attempts alternative files.
        """
        all_embeddings = []
        labels = []
        skipped_files = 0
        processed_files = set()
        
        # Get all .npy files in the directory
        all_files = list(Path(embeddings_dir).glob('*.npy'))
        
        for emb_file in all_files:
            if len(all_embeddings) >= len(all_files):
                break  # Don't process more files than available
                
            if emb_file in processed_files:
                continue
                
            attempts = 0
            current_file = emb_file
            
            while attempts < max_attempts:
                try:
                    # Try to load the file
                    embeddings = np.load(current_file)
                    
                    # Check for NaN values or empty arrays
                    if len(embeddings) == 0 or np.isnan(embeddings).any():
                        skipped_files += 1
                        print(f"Skipping {current_file.name} (attempt {attempts + 1}/{max_attempts}): Contains NaN values")
                        
                        # Try to find an alternative file
                        remaining_files = [f for f in all_files if f not in processed_files and f != current_file]
                        if remaining_files:
                            current_file = remaining_files[0]
                            attempts += 1
                            continue
                        else:
                            break  # No more files to try
                    
                    # Process the file if it's valid
                    agg_embedding = extract_aggregated_features(embeddings, use_attention=True)
                    
                    if agg_embedding is not None and not np.isnan(agg_embedding).any():
                        all_embeddings.append(agg_embedding)
                        labels.append(1 if 'Positive' in str(current_file) else 0)
                        processed_files.add(current_file)
                        break
                    else:
                        skipped_files += 1
                        
                except Exception as e:
                    print(f"Error processing {current_file}: {str(e)}")
                    skipped_files += 1
                    
                # If we get here, there was an issue with the current file
                attempts += 1
                remaining_files = [f for f in all_files if f not in processed_files and f != current_file]
                if remaining_files:
                    current_file = remaining_files[0]
                else:
                    break  # No more files to try
                
        if skipped_files > 0:
            print(f"Warning: Skipped {skipped_files} files due to NaN values or processing errors")
            
        if not all_embeddings:
            raise ValueError(f"No valid embeddings found in {embeddings_dir}")
            
        return np.array(all_embeddings), np.array(labels)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embeddings
    print("Loading positive class embeddings...")
    X_pos, y_pos = load_embeddings(positive_dir)
    print(f"Loaded {len(X_pos)} positive samples")
    
    print("Loading negative class embeddings...")
    X_neg, y_neg = load_embeddings(negative_dir)
    print(f"Loaded {len(X_neg)} negative samples")
    
    # Combine data
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    
    # Final check for NaN values
    nan_mask = np.isnan(X).any(axis=1)
    if np.any(nan_mask):
        print(f"Warning: Found {np.sum(nan_mask)} samples with NaN values. Removing them...")
        X = X[~nan_mask]
        y = y[~nan_mask]
    
    print(f"Final dataset shape: {X.shape}, {y.shape}")
    
    # Dimensionality reduction for visualization
    print("Reducing dimensions for visualization...")
    
    # Use PCA first to reduce to 50 dimensions
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    # Then use t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plot t-SNE visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y,
        palette={0: 'blue', 1: 'red'},
        alpha=0.7,
        s=50
    )
    plt.title('t-SNE Visualization of Audio Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Class', labels=['Negative', 'Positive'])
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()
    
    # Calculate and print statistics
    print("\nClass Statistics:")
    print(f"Positive samples: {len(X_pos)}")
    print(f"Negative samples: {len(X_neg)}")
    print(f"\nMean embedding norm:")
    print(f"  Positive: {np.mean(np.linalg.norm(X_pos, axis=1)):.4f}")
    print(f"  Negative: {np.mean(np.linalg.norm(X_neg, axis=1)):.4f}")
    
    # Save results to file
    with open(os.path.join(output_dir, 'comparison_results.txt'), 'w') as f:
        f.write("Class Comparison Results\n")
        f.write("======================\n\n")
        f.write(f"Positive samples: {len(X_pos)}\n")
        f.write(f"Negative samples: {len(X_neg)}\n\n")
        f.write("Mean embedding norms:\n")
        f.write(f"  Positive: {np.mean(np.linalg.norm(X_pos, axis=1)):.4f}\n")
        f.write(f"  Negative: {np.mean(np.linalg.norm(X_neg, axis=1)):.4f}\n")
    
    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    # Run the experiment first
    # run_experiment()
    
    # Then compare the embeddings
    print("\n" + "="*50)
    print("COMPARING POSITIVE AND NEGATIVE CLASSES")
    print("="*50)
    
    positive_dir = "data/covid_positive/raw_embeddings"
    negative_dir = "data/non_covid_negative/raw_embeddings"
    
    compare_embeddings(positive_dir, negative_dir)
