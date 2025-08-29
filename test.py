from google.cloud import aiplatform
from google.oauth2 import service_account
import json
import base64
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

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

# Example audio file path
audio_file_path = "test2.wav"

# Function to get embeddings
def get_hear_embeddings(audio_file_path):
    # Read the audio file as bytes
    with open(audio_file_path, "rb") as f:
        audio_content = f.read()
    
    try:
        # Using input_bytes with base64 encoding directly in the instance
        instances = [{
            "input_bytes": base64.b64encode(audio_content).decode("utf-8")
        }]
        response = endpoint.predict(instances=instances)
        
        # Convert the response to a numpy array
        if isinstance(response.predictions[0], dict):
            # If the prediction is a dictionary, extract the values
            embeddings = np.array(list(response.predictions[0].values()))
        else:
            embeddings = np.array(response.predictions[0])
            
        print(f"Raw embedding shape from API: {embeddings.shape}")
        print(f"First few values: {embeddings[:5]}")
        return embeddings
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def predict_audio(model, audio_file_path):
    """Predict the class of an audio file using the trained model."""
    # Get embeddings for the audio file
    embeddings = get_hear_embeddings(audio_file_path)
    
    if embeddings is not None:
        # Reshape to 2D array (1 sample, n_features)
        embeddings = embeddings.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(embeddings)
        probabilities = model.predict_proba(embeddings)
        
        print(f"\nPrediction for {audio_file_path}:")
        print(f"Predicted class: {prediction[0]}")
        print(f"Class probabilities: {probabilities[0]}")
        
        return prediction[0]
    else:
        print("Failed to get embeddings for prediction")
        return None

def load_embeddings_from_public_dataset(folder_path):
    """Load all embeddings from the public dataset, handling different formats."""
    import numpy as np
    from pathlib import Path
    
    folder = Path(folder_path)
    embeddings = []
    
    # Load first 1000 embeddings (adjust as needed)
    for file_path in list(folder.glob('*.npy'))[:1000]:  # Limiting to first 1000 for speed
        try:
            # Load with allow_pickle=True
            data = np.load(file_path, allow_pickle=True)
            
            # Handle different data formats
            if isinstance(data, np.ndarray):
                if data.dtype == np.object_:
                    # If it's an object array, try to extract the first element
                    data = data.item()
                
                if isinstance(data, dict):
                    # If it's a dictionary, try to find the embedding data
                    if 'embedding' in data:
                        emb = data['embedding']
                    elif 'features' in data:
                        emb = data['features']
                    else:
                        # Try to use the first value that looks like an embedding
                        for v in data.values():
                            if isinstance(v, (np.ndarray, list)) and len(v) > 10:  # Simple check for embedding-like data
                                emb = v
                                break
                        else:
                            emb = None
                    
                    if emb is not None:
                        emb = np.array(emb, dtype=np.float32)
                else:
                    # It's already a numpy array
                    emb = np.array(data, dtype=np.float32)
                
                # Ensure it's 1D and has the right shape
                if emb is not None and len(emb) > 0:
                    emb = emb.flatten()  # Ensure it's 1D
                    if len(emb) == 512:  # Expected embedding size
                        embeddings.append(emb)
                    else:
                        print(f"Skipping {file_path.name}: unexpected shape {emb.shape}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    if not embeddings:
        raise ValueError("No valid embeddings found in the dataset")
        
    return np.array(embeddings)

def train_model(embeddings, labels):
    """Train and compare multiple models on the embeddings."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Define models to compare
    models = {
        "Support Vector Machine (linear)": SVC(kernel='linear', random_state=42, probability=True),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=128, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=128, random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=1000)
    }
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    trained_models = {}
    results = {}
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate scores
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        
        # Store results
        trained_models[name] = model
        results[name] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score
        }
        
        # Save the model
        model_path = os.path.join('models', f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")
        
        # Print results
        print(f"{name} - Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
        
        # Track best model
        if test_score > best_score:
            best_score = test_score
            best_model = model
    
    # Print summary
    print("\n=== Model Comparison ===")
    for name, result in results.items():
        print(f"{name}: Test Accuracy = {result['test_score']:.4f}")
    
    best_model_name = [k for k, v in results.items() if v['test_score'] == best_score][0]
    print(f"\nBest model: {best_model_name}")
    print(f"Best test accuracy: {best_score:.4f}")
    
    # Save the best model separately
    best_model_path = os.path.join('models', 'best_model.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"\nSaved best model to {best_model_path}")
    
    return best_model, X_test_scaled, y_test, results

if __name__ == "__main__":
    # Load embeddings from public dataset
    print("Loading embeddings from public dataset...")
    try:
        embeddings = load_embeddings_from_public_dataset("public_dataset/embeddings")
        print(f"Loaded {len(embeddings)} embeddings with shape {embeddings[0].shape}")
        
        # Create dummy labels (replace with your actual labels)
        # For binary classification (0 or 1)
        labels = np.random.randint(0, 2, size=len(embeddings))
        
        # Train the model
        print("\nTraining model...")
        model, X_test, y_test, results = train_model(embeddings, labels)
        
        label_names = {0: "Non-COVID", 1: "COVID"}
        
        # Test with your audio file
        test_audio = "/Users/ronel/Downloads/dev/templates/hearpython/public_dataset_non_covid/non_covid/0a1b4119-cc22-4884-8f0f-34e8207c31d1.wav"
        if os.path.exists(test_audio):
            print("\nMaking prediction on test audio...")
            prediction = predict_audio(model, test_audio)
            if prediction is not None:
                print(f"Final prediction for {test_audio}: Class {label_names[prediction]}")
        else:
            print(f"\nTest audio file not found: {test_audio}")
            
    except Exception as e:
        print(f"An error occurred: {e}")