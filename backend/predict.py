import pandas as pd
import re
import numpy as np
from urllib.parse import urlparse
import math
import joblib
from typing import Dict, Any, Tuple

# Function to extract URL features (copied from trainModel.py)
def extract_url_features(url: str) -> Dict[str, Any]:
    features = {}
    
    # Raw URL
    features['raw_url'] = url
    
    # Parse the URL
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
    except:
        parsed_url = None
        domain = ""
    
    # Raw domain
    features['raw_domain'] = domain
    
    # Check if HTTPS
    features['is_https'] = 1 if url.startswith('https://') else 0
    
    # URL Length
    features['url_length'] = len(url)
    
    # Number of Subdomains
    subdomains = domain.split('.')
    features['num_subdomains'] = len(subdomains) - 1 if len(subdomains) > 1 else 0
    
    # Presence of @ Symbol
    features['has_at_symbol'] = 1 if '@' in url else 0
    
    # Number of Special Characters
    special_chars = re.findall(r'[^a-zA-Z0-9.]', url)
    features['num_special_chars'] = len(special_chars)
    
    # Presence of Suspicious Keywords
    suspicious_keywords = ['login', 'secure', 'account', 'banking', 'confirm', 'verify', 'paypal', 'password']
    features['has_suspicious_keywords'] = 0
    for keyword in suspicious_keywords:
        if keyword in url.lower():
            features['has_suspicious_keywords'] = 1
            break
    
    # Ratio of Digits to Letters
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    features['digit_letter_ratio'] = digits / letters if letters > 0 else 0
    
    # Entropy of URL String
    if url:
        prob = [float(url.count(c)) / len(url) for c in set(url)]
        features['entropy'] = -sum(p * math.log(p, 2) for p in prob)
    else:
        features['entropy'] = 0
    
    # Length of Domain Name
    features['domain_length'] = len(domain)
    
    # Count of URL Parameters
    features['num_parameters'] = url.count('&') + 1 if '?' in url else 0
    
    # Number of Hyphens in Domain
    features['num_hyphens_in_domain'] = domain.count('-')
    
    # Top-Level Domain (TLD)
    features['tld'] = domain.split('.')[-1] if domain and len(domain.split('.')) > 1 else ""
    
    # Is Domain Similar to Popular Brand
    popular_brands = ['google', 'facebook', 'amazon', 'apple', 'microsoft', 'paypal', 'yahoo', 'instagram']
    features['similar_to_brand'] = 0
    domain_without_tld = '.'.join(domain.split('.')[:-1]) if domain and len(domain.split('.')) > 1 else domain
    for brand in popular_brands:
        if brand in domain_without_tld.lower() and brand != domain_without_tld.lower():
            features['similar_to_brand'] = 1
            break
    
    # URL Path Length
    features['path_length'] = len(parsed_url.path) if parsed_url and parsed_url.path else 0
    
    return features

def predict_url(url: str) -> Dict[str, Any]:
    """
    Predicts whether a URL is legitimate or phishing using all available models.
    Returns the best prediction based on confidence scores.
    
    Args:
        url: The URL to predict
        
    Returns:
        Dictionary containing prediction results and confidence scores
    """
    # Load models
    try:
        # rf_model = joblib.load('backend/models/random_forest_model.pkl')
        xgb_model = joblib.load('backend/models/xgboost_model.pkl')
        svm_model = joblib.load('backend/models/svm_model.pkl')
        lr_model = joblib.load('backend/models/logistic_regression_model.pkl')
    except:
        # Try alternative path
        # rf_model = joblib.load('models/random_forest_model.pkl')
        xgb_model = joblib.load('models/xgboost_model.pkl')
        svm_model = joblib.load('models/svm_model.pkl')
        lr_model = joblib.load('models/logistic_regression_model.pkl')
    
    # Extract features from the URL
    features = extract_url_features(url)
    
    # Prepare features for prediction (drop non-numeric columns)
    X = pd.DataFrame([features])
    X_pred = X.drop(['raw_url', 'raw_domain', 'tld'], axis=1, errors='ignore')
    
    # Make predictions with each model
    # rf_pred_proba = rf_model.predict_proba(X_pred)[0]
    xgb_pred_proba = xgb_model.predict_proba(X_pred)[0]
    svm_pred_proba = svm_model.predict_proba(X_pred)[0]
    lr_pred_proba = lr_model.predict_proba(X_pred)[0]
    
    # Get prediction labels (0 = phishing, 1 = legitimate)
    # rf_pred = rf_model.predict(X_pred)[0]
    xgb_pred = xgb_model.predict(X_pred)[0]
    svm_pred = svm_model.predict(X_pred)[0]
    lr_pred = lr_model.predict(X_pred)[0]
    
    # Calculate confidence scores (probability of the predicted class)
    # rf_confidence = rf_pred_proba[1] if rf_pred == 1 else rf_pred_proba[0]
    xgb_confidence = xgb_pred_proba[1] if xgb_pred == 1 else xgb_pred_proba[0]
    svm_confidence = svm_pred_proba[1] if svm_pred == 1 else svm_pred_proba[0]
    lr_confidence = lr_pred_proba[1] if lr_pred == 1 else lr_pred_proba[0]
    
    # Determine the most confident prediction
    confidences = [
        # ("Random Forest", rf_confidence, rf_pred),
        ("XGBoost", xgb_confidence, xgb_pred),
        ("SVM", svm_confidence, svm_pred),
        ("Logistic Regression", lr_confidence, lr_pred)
    ]
    
    best_model, best_confidence, best_pred = max(confidences, key=lambda x: x[1])
    
    # Convert prediction to string for better readability
    prediction_str = "legitimate" if best_pred == 1 else "phishing"
    
    # Prepare and return the result
    result = {
        "url": url,
        "prediction": prediction_str,
        "confidence": round(float(best_confidence), 4),
        "best_model": best_model,
        "model_predictions": {
            # "random_forest": {"prediction": "legitimate" if rf_pred == 1 else "phishing", "confidence": round(float(rf_confidence), 4)},
            "xgboost": {"prediction": "legitimate" if xgb_pred == 1 else "phishing", "confidence": round(float(xgb_confidence), 4)},
            "svm": {"prediction": "legitimate" if svm_pred == 1 else "phishing", "confidence": round(float(svm_confidence), 4)},
            "logistic_regression": {"prediction": "legitimate" if lr_pred == 1 else "phishing", "confidence": round(float(lr_confidence), 4)}
        }
    }
    
    return result

# For testing
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "https://secure-banking.login-account.com/verify",
        "https://www.amazon.com/dp/B084DWCZY6/"
    ]
    
    print("Testing URL prediction with all models:\n")
    
    for test_url in test_urls:
        result = predict_url(test_url)
        print(f"URL: {result['url']}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']})")
        print(f"Best model: {result['best_model']}")
        print("\nAll model predictions:")
        for model, data in result['model_predictions'].items():
            print(f"  {model.capitalize()}: {data['prediction']} (Confidence: {data['confidence']})")
        print("\n" + "-"*50 + "\n")
