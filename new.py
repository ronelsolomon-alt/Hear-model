from google.cloud import aiplatform
from google.oauth2 import service_account
import json
import base64
import numpy as np
import soundfile as sf
import joblib
import os
from pathlib import Path
import joblib
import tempfile
from moviepy.editor import AudioFileClip
import audioop
from pydub import AudioSegment
from scipy.io import wavfile

# Load your service account credentials
credentials_path = 'test.json'
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Initialize the Vertex AI client
aiplatform.init(
    project=credentials.project_id,
    location='us-central1',
    credentials=credentials
)

# Your endpoint details
endpoint_id = '5860664157371629568'
endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{credentials.project_id}/locations/us-central1/endpoints/{endpoint_id}"
)

def convert_audio_to_wav(audio_path):
    """Convert audio file to WAV format and return the path to the temporary file."""
    try:
        # Create a temporary file for the converted audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Load the audio file
            audio = AudioFileClip(audio_path)
            # Write the audio to the temporary file in WAV format
            audio.write_audiofile(temp_audio.name, codec='pcm_s16le', verbose=False, logger=None, fps=16000)  # Standard sample rate for speech
            return temp_audio.name
    except Exception as e:
        print(f"Error converting {audio_path} to WAV: {str(e)}")
        return None

def get_hear_embeddings(audio_input, chunk_duration=2000):
    """Get embeddings for an audio file or numpy array using the HEAR model.
    
    Args:
        audio_input: Either a path to an audio file or a numpy array containing audio samples
        chunk_duration: Duration of each chunk in milliseconds (default: 2000ms = 2s)
    Returns:
        Combined embeddings from all chunks, or None if processing fails
    """
    temp_audio = None
    output_files = []
    all_embeddings = []
    
    try:
        # Handle numpy array input
        if isinstance(audio_input, np.ndarray):
            # Convert numpy array to temporary WAV file
            temp_audio = "temp_audio.wav"
            sf.write(temp_audio, audio_input, 16000)  # Assuming 16kHz sample rate
            audio_file_path = temp_audio
        else:
            # Handle file path input
            audio_file_path = audio_input
            # Check if file needs conversion (MP3 or MP4)
            if isinstance(audio_file_path, str) and audio_file_path.lower().endswith(('.mp3', '.mp4')):
                temp_audio = convert_audio_to_wav(audio_file_path)
                if not temp_audio:
                    return None
                audio_file_path = temp_audio
        
        # Process with pydub
        try:
            audio = AudioSegment.from_file(audio_file_path)
            duration_ms = len(audio)
            
            # Process in chunks
            for start_ms in range(0, duration_ms, chunk_duration):
                end_ms = min(start_ms + chunk_duration, duration_ms)
                chunk = audio[start_ms:end_ms]
                
                # Pad if this is the last chunk and it's too short
                if len(chunk) < chunk_duration:
                    silence = AudioSegment.silent(duration=chunk_duration - len(chunk))
                    chunk = chunk + silence
                
                # Create a temporary file for this chunk
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    output_file = f.name
                    output_files.append(output_file)
                
                # Export chunk to WAV format
                chunk.export(output_file, format="wav",
                           parameters=['-ac', '1', '-ar', '16000',
                                     '-acodec', 'pcm_s16le'])
                
                # Read the chunk content
                with open(output_file, "rb") as f:
                    audio_content = f.read()
                
                # Prepare and send the request for this chunk
                instances = [{"input_bytes": base64.b64encode(audio_content).decode("utf-8")}]
                
                try:
                    response = endpoint.predict(instances=instances)
                    
                    if not hasattr(response, 'predictions') or not response.predictions:
                        print(f"No predictions received for chunk starting at {start_ms}ms")
                        continue
                        
                    prediction = response.predictions[0]
                    
                    if isinstance(prediction, dict):
                        if 'error' in prediction:
                            print(f"Error from HEAR endpoint: {prediction}")
                            continue
                        # If prediction is a dictionary, convert values to array
                        embedding = np.array(list(prediction.values()), dtype=np.float32)
                    else:
                        # If prediction is already an array-like
                        embedding = np.array(prediction, dtype=np.float32)
                    
                    # Flatten and ensure 512 dimensions
                    embedding = embedding.flatten()
                    if len(embedding) < 512:
                        embedding = np.pad(embedding, (0, 512 - len(embedding)))
                    elif len(embedding) > 512:
                        embedding = embedding[:512]
                    
                    all_embeddings.append(embedding)
                    print(f"Processed chunk {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
                    
                except Exception as e:
                    print(f"Error processing chunk starting at {start_ms}ms: {str(e)}")
        
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {str(e)}")
            return None
            
        if not all_embeddings:
            print("No valid embeddings were generated")
            return None
            
        print(f"Processed {len(all_embeddings)} chunks from {audio_file_path}")
        
        # Take max across all chunk embeddings to get the most prominent features
        max_embedding = np.max(all_embeddings, axis=0)
        
        # Return as 2D array with shape (1, 512)
        return max_embedding.reshape(1, -1)
        
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up all temporary files
        try:
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
            for f in output_files:
                if os.path.exists(f):
                    os.remove(f)
        except Exception as e:
            print(f"Warning: Error cleaning up temporary files: {str(e)}")

def load_models(models_dir='models'):
    """Load all trained models from the models directory."""
    models = {}
    model_files = list(Path(models_dir).glob('*.joblib'))
    
    for model_file in model_files:
        try:
            model_name = model_file.stem
            if model_name != 'best_model':  # Skip best_model as it's a duplicate
                models[model_name] = joblib.load(model_file)
                print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_file}: {str(e)}")
    
    return models

def predict_with_models(audio_path, models):
    """Make predictions using all loaded models."""
    # Get embeddings for the audio file
    embeddings = get_hear_embeddings(audio_path)
    if embeddings is None:
        print("Failed to get embeddings")
        return None
    
    predictions = {}
    for model_name, model in models.items():
        # Skip the scaler as it's not a predictive model
        if 'scaler' in model_name.lower():
            print(f"Skipping {model_name} as it's not a predictive model")
            continue
            
        try:
            # Handle models that are saved as dictionaries with metadata
            if isinstance(model, dict) and 'model' in model:
                print(f"Model {model_name} is a dictionary with metadata, extracting model...")
                # Get the actual model from the dictionary
                model_to_use = model['model']
                # Get any additional metadata
                model_metadata = {k: v for k, v in model.items() if k != 'model'}
                print(f"Model metadata: {list(model_metadata.keys())}")
            else:
                model_to_use = model
                model_metadata = {}
            
            # Make prediction
            try:
                # Make prediction
                prediction = model_to_use.predict(embeddings)
                print(f"\nDebug - Model: {model_name}")
                print(f"Prediction type: {type(prediction)}")
                print(f"Prediction content: {prediction}")
                
                # Initialize default values
                class_idx = 0
                confidence = 0.5
                proba = [0.5, 0.5]  # Default probabilities
                
                # Get prediction probabilities if available
                try:
                    if hasattr(model_to_use, 'predict_proba'):
                        proba = model_to_use.predict_proba(embeddings)[0]
                        confidence = float(max(proba))
                        class_idx = int(np.argmax(proba))
                        print(f"Probabilities: {proba}")
                    else:
                        # For models without predict_proba, just use the prediction
                        if hasattr(prediction, '__getitem__'):
                            class_idx = int(prediction[0])
                        else:
                            class_idx = int(prediction)
                        confidence = 1.0  # Default confidence since we don't have probabilities
                except Exception as proba_error:
                    print(f"Warning getting probabilities: {str(proba_error)}")
                    if hasattr(prediction, '__getitem__'):
                        class_idx = int(prediction[0])
                    else:
                        class_idx = int(prediction)
                
                # Store the prediction with consistent format
                predictions[model_name] = {
                    'class': 'COVID' if class_idx == 1 else 'Non-COVID',
                    'confidence': confidence,
                    'probabilities': {
                        'COVID': float(proba[1]) if isinstance(proba, (list, np.ndarray)) and len(proba) > 1 else (confidence if class_idx == 1 else 1 - confidence),
                        'Non-COVID': float(proba[0]) if isinstance(proba, (list, np.ndarray)) and len(proba) > 0 else (1 - confidence if class_idx == 1 else confidence)
                    }
                }
                
            except Exception as e:
                print(f"Error making prediction with {model_name}: {str(e)}")
                predictions[model_name] = {
                    'error': str(e),
                    'class': 'Error',
                    'confidence': 0.0,
                    'probabilities': {'COVID': 0.0, 'Non-COVID': 0.0}
                }
            
        except Exception as e:
            print(f"Error predicting with {model_name}: {str(e)}")
    
    return predictions

if __name__ == "__main__":
    # Load all models
    print("Loading models...")
    models = load_models()
    
    if not models:
        print("No models found in the models directory")
        exit(1)
    
    # Path to test audio file test2.wav
    # /Users/ronel/Downloads/dev/templates/hearpython/public_dataset_non_covid/covid/0a03da19-eb19-4f51-9860-78ad95fa8cb5.wav covid
    # /Users/ronel/Downloads/dev/templates/hearpython/public_dataset_non_covid/non_covid/0a1b4119-cc22-4884-8f0f-34e8207c31d1.wav non covid
    test_audio = "/Users/ronel/Downloads/dev/templates/hearpython/public_dataset_non_covid/non_covid/0a1b4119-cc22-4884-8f0f-34e8207c31d1.wav"
    if not os.path.exists(test_audio):
        print(f"Test audio file not found: {test_audio}")
        exit(1)
    
    # Make predictions
    print(f"\nMaking predictions for {test_audio}...")
    predictions = predict_with_models(test_audio, models)
    
    if predictions:
        # Print results
        print("=== Prediction Results ===\n")
        for model_name, pred in predictions.items():
            if 'error' in pred:
                print(f"{model_name}: Error - {pred['error']}")
                continue
                
            print(f"{model_name}:")
            print(f"  Prediction: {pred['class']}")
            print(f"  Confidence: {pred['confidence']*100:.2f}%")
            print(f"  Probabilities:")
            print(f"    Non-COVID: {pred['probabilities']['Non-COVID']*100:.2f}%")
            print(f"    COVID: {pred['probabilities']['COVID']*100:.2f}%")
            print()
    else:
        print("No predictions were made due to errors.")