import pickle
import numpy as np

def load_models():
    """Load the pre-trained models from disk"""
    # Load Take/Keep classifier
    with open('take_keep_classifier.pkl', 'rb') as f:
        take_keep_model_data = pickle.load(f)
    
    # Load Toxic Gas classifier
    with open('toxic_gas_model.pkl', 'rb') as f:
        toxic_gas_model_data = pickle.load(f)
    
    return take_keep_model_data, toxic_gas_model_data

def predict_classification(input_raw):
    """
    Predicts both the action (Take/Keep) and Gas type (Normal/Toxic)
    from a given RAW sensor input vector using pre-trained SVM models and PCA.
    
    Args:
        input_raw: A list or array with 3 values representing [gas_value, distance, weight]
                  for Take/Keep model or [fill_level, weight, gas_concentration] for Toxic gas
    
    Returns:
        A dictionary containing the predictions and confidence scores
    """
    # Convert input to numpy array if it's not already
    input_raw = np.array(input_raw).reshape(1, -1)
    
    # Load models
    take_keep_model_data, toxic_gas_model_data = load_models()
    
    # Process for Take/Keep model
    take_keep_scaler = take_keep_model_data['scaler']
    take_keep_pca = take_keep_model_data['pca']
    take_keep_model = take_keep_model_data['model']
    
    # Process for Toxic Gas model
    toxic_gas_mu = toxic_gas_model_data['mu']
    toxic_gas_pca = toxic_gas_model_data['pca']
    toxic_gas_model = toxic_gas_model_data['model']
    
    # Preprocess for Take/Keep
    input_normalized = take_keep_scaler.transform(input_raw)
    input_pca_take_keep = take_keep_pca.transform(input_normalized)
    
    # Preprocess for Toxic Gas
    input_centered = input_raw - toxic_gas_mu
    input_pca_toxic = toxic_gas_pca.transform(input_raw)
    
    # Make predictions
    action_prob = take_keep_model.predict_proba(input_pca_take_keep)[0]
    gas_prob = toxic_gas_model.predict_proba(input_pca_toxic)[0]
    
    action_label = take_keep_model.predict(input_pca_take_keep)[0]
    gas_label = toxic_gas_model.predict(input_pca_toxic)[0]
    
    # Convert predictions to string labels
    action_str = "Take" if action_label == 1 else "Keep"
    gas_str = "Toxic" if gas_label == 1 else "Normal"
    
    # Calculate confidence
    action_confidence = action_prob[action_label] * 100
    gas_confidence = gas_prob[gas_label] * 100
    
    # Build result dictionary
    result = {
        "action": {
            "label": action_str,
            "confidence": round(action_confidence, 2)
        },
        "gas": {
            "label": gas_str,
            "confidence": round(gas_confidence, 2)
        },
        "message": f"Action: {action_str} (Confidence: {action_confidence:.2f}%)\nGas: {gas_str} (Confidence: {gas_confidence:.2f}%)"
    }
    
    return result