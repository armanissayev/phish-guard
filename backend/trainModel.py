import pandas as pd
import re
import numpy as np
from urllib.parse import urlparse
import math
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import joblib
import glob
import os

# Load the dataset
file_paths = glob.glob(os.path.join("data", "*.csv"))
dataframes = [pd.read_csv(file) for file in file_paths]
df = pd.concat(dataframes, ignore_index=True)

# Display the first few rows of the dataset
print(df.head())

# Handle www bias: Create additional rows for URLs with/without www
url_column = 'url'  # Change this if your URL column has a different name
if url_column not in df.columns:
    # Try to find a column that might contain URLs
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains('http').any():
            url_column = col
            print(f"Using column '{url_column}' as the URL column")
            break

# Create a copy of the dataframe to add modified URLs
additional_rows = []

# Process each URL to create versions with/without www
for index, row in df.iterrows():
    url = row[url_column]
    url_type = row['type']
    
    # Parse the URL
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        scheme = parsed_url.scheme
        path = parsed_url.path
        params = parsed_url.params
        query = parsed_url.query
        fragment = parsed_url.fragment
        
        # Check if domain starts with www.
        if domain.startswith('www.'):
            # Create version without www
            new_domain = domain[4:]  # Remove 'www.'
            # Reconstruct the URL properly
            new_url = urlparse(url)._replace(netloc=new_domain).geturl()
            # print(url, " -> ", new_url)
            # print(new_url)
            
            # Add to additional rows
            new_row = row.copy()
            new_row[url_column] = new_url
            additional_rows.append(new_row)
        else:
            # Create version with www
            new_domain = 'www.' + domain
            # Reconstruct the URL properly
            new_url = urlparse(url)._replace(netloc=new_domain).geturl()
            # print(new_url)
            
            # Add to additional rows
            new_row = row.copy()
            new_row[url_column] = new_url
            additional_rows.append(new_row)
    except:
        # Skip if URL parsing fails
        continue

# Add the additional rows to the dataframe
if additional_rows:
    additional_df = pd.DataFrame(additional_rows)
    df = pd.concat([df, additional_df], ignore_index=True)
    print(f"\nAdded {len(additional_rows)} URLs with www/non-www variants")
    print(f"New dataset size: {len(df)} rows")

# Shuffle the rows of the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("\nDataFrame after shuffling:")
print(df.head())

# Function to extract URL features
def extract_url_features(url):
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

# Extract features for each URL in the dataset
# Assuming the URL column is named 'url' - adjust if it's different
url_column = 'url'  # Change this if your URL column has a different name

# Check if the URL column exists
if url_column not in df.columns:
    # Try to find a column that might contain URLs
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains('http').any():
            url_column = col
            print(f"Using column '{url_column}' as the URL column")
            break
    else:
        raise ValueError("Could not find a column containing URLs")

# Extract features
features_list = []
for url in df[url_column]:
    features_list.append(extract_url_features(url))

# Create a DataFrame from the extracted features
X = pd.DataFrame(features_list)
y = df['type'].map({'legitimate': 1, 'phishing': 0})

# Display the first few rows of the features table
print("\nExtracted URL Features:")
print(X.head())

# Prepare the data for model training
# Drop non-numeric columns that can't be used directly for training
X_train = X.drop(['raw_url', 'raw_domain', 'tld'], axis=1, errors='ignore')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Train SVM model
print("\nTraining SVM model...")
# Using LinearSVC instead of SVC for much better performance on large datasets
# LinearSVC is much faster as it implements a linear kernel with optimized algorithms
base_svm = LinearSVC(dual="auto", random_state=42, max_iter=1000)
# Wrap with CalibratedClassifierCV to get probability estimates (equivalent to SVC(probability=True))
svm_model = CalibratedClassifierCV(base_svm, cv=3)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, svm_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, lr_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, xgb_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

# Compare models
print("\nModel Comparison:")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Save the best model
best_accuracy = max(rf_accuracy, svm_accuracy, lr_accuracy, xgb_accuracy)
if best_accuracy == rf_accuracy:
    best_model = rf_model
    model_name = "random_forest_model.pkl"
elif best_accuracy == svm_accuracy:
    best_model = svm_model
    model_name = "svm_model.pkl"
elif best_accuracy == lr_accuracy:
    best_model = lr_model
    model_name = "logistic_regression_model.pkl"
else:
    best_model = xgb_model
    model_name = "xgboost_model.pkl"

print(f"\nSaving the best model ({model_name}) with accuracy: {best_accuracy:.4f}")
joblib.dump(best_model, f"models/{model_name}")

# Also save all models for future use
print("\nSaving all models...")
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(lr_model, "models/logistic_regression_model.pkl")
joblib.dump(xgb_model, "models/xgboost_model.pkl")

print("\nTraining and evaluation completed.")
 
