#!/usr/bin/env python3
"""
Extract and serialize models from the HouseRental Jupyter notebook
This script recreates the data processing and model training pipeline
and saves the trained models as pickle files for use in the Flask app.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """
    Create sample data since we don't have the original CSV file
    This mirrors the structure from the notebook
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data with similar structure to the original
    data = {
        'listing_id': range(1, n_samples + 1),
        'listing_category': np.random.choice(['House', 'Apartment', 'Condo'], n_samples),
        'bath_count': np.random.randint(1, 4, n_samples),
        'bed_count': np.random.randint(1, 5, n_samples),
        'payment_currency': np.random.choice(['USD', 'EUR'], n_samples),
        'service_fee_applicable': np.random.choice([True, False], n_samples),
        'image_available': np.random.choice([True, False], n_samples),
        'pet_policy': np.random.choice(['Allowed', 'Not Allowed', 'Conditional'], n_samples),
        'payment_schedule': np.random.choice(['Monthly', 'Weekly'], n_samples),
        'area_in_sqft': np.random.normal(1200, 400, n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
        'region_code': np.random.choice(['NY', 'CA', 'IL', 'TX'], n_samples),
        'geo_lat': np.random.uniform(25, 48, n_samples),
        'geo_lon': np.random.uniform(-125, -70, n_samples),
        'data_provider': np.random.choice(['Provider1', 'Provider2'], n_samples)
    }
    
    # Create monthly_rent based on features (realistic relationship)
    df = pd.DataFrame(data)
    base_rent = (
        df['bed_count'] * 400 +
        df['bath_count'] * 200 +
        df['area_in_sqft'] * 1.5 +
        np.random.normal(0, 200, n_samples)
    )
    df['monthly_rent'] = np.clip(base_rent, 500, 5000)
    
    return df

def preprocess_data(df):
    """
    Preprocess data following the same steps as in the notebook
    """
    print("Starting data preprocessing...")
    
    # Drop text columns if they exist
    text_cols = ['headline', 'street_address', 'formatted_rent_text', 'included_features']
    df = df.drop(columns=[col for col in text_cols if col in df.columns], errors='ignore')
    
    # Handle missing values
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    
    # Fill numerical missing values with median
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Group rare categories (following notebook logic)
    categorical_to_group = ['listing_category', 'payment_currency', 'payment_schedule', 
                           'data_provider', 'pet_policy', 'city', 'region_code']
    
    for col in categorical_to_group:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            common = freq[freq >= 0.05].index
            df.loc[:, col] = df[col].where(df[col].isin(common), 'Other')
    
    # One-hot encoding
    encode_cols = ['listing_category', 'payment_currency', 'service_fee_applicable', 
                   'image_available', 'payment_schedule', 'data_provider', 
                   'pet_policy', 'city', 'region_code']
    
    df_encoded = pd.get_dummies(df, columns=[col for col in encode_cols if col in df.columns], drop_first=True)
    
    # Remove outliers using IQR method (from notebook)
    if 'monthly_rent' in df_encoded.columns:
        Q1 = df_encoded['monthly_rent'].quantile(0.25)
        Q3 = df_encoded['monthly_rent'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_encoded = df_encoded[(df_encoded['monthly_rent'] >= lower_bound) & 
                               (df_encoded['monthly_rent'] <= upper_bound)]
    
    # Scale numerical features
    num_cols = df_encoded.select_dtypes(include=['number']).columns
    num_cols = num_cols.drop(['monthly_rent', 'listing_id'], errors='ignore')
    
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    
    # MinMax scaling
    minmax_scaler = MinMaxScaler()
    df_encoded[num_cols] = minmax_scaler.fit_transform(df_encoded[num_cols])
    
    # Create HighRent classification target
    median_rent = df_encoded['monthly_rent'].median()
    df_encoded['HighRent'] = (df_encoded['monthly_rent'] > median_rent).astype(int)
    
    print(f"Data preprocessing completed. Final shape: {df_encoded.shape}")
    
    return df_encoded, scaler, minmax_scaler

def train_and_save_models():
    """
    Train models and save them as pickle files
    """
    print("Creating sample data...")
    df = create_sample_data()
    
    # Preprocess data
    df_processed, scaler, minmax_scaler = preprocess_data(df)
    
    # Prepare features
    feature_cols = [col for col in df_processed.columns 
                   if col not in ['monthly_rent', 'HighRent', 'listing_id']]
    X = df_processed[feature_cols]
    
    # Remove low variance features (from notebook)
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    X_final = pd.DataFrame(X_selected, columns=selected_features)
    
    print(f"Selected {len(selected_features)} features after variance threshold")
    
    # Prepare targets
    y_reg = df_processed['monthly_rent']
    y_clf = df_processed['HighRent']
    
    # Split data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_final, y_reg, test_size=0.3, random_state=42
    )
    
    # Split data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_final, y_clf, test_size=0.3, random_state=42, stratify=y_clf
    )
    
    # Dictionary to store all models and preprocessing objects
    models_dict = {}
    
    print("\nTraining models...")
    
    # 1. Linear Regression
    print("Training Linear Regression...")
    linreg = LinearRegression()
    linreg.fit(X_train_reg, y_train_reg)
    models_dict['linear_regression'] = linreg
    
    # 2. Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_reg, y_train_reg)
    models_dict['random_forest_regressor'] = rf_reg
    
    # 3. XGBoost Regressor
    print("Training XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_reg.fit(X_train_reg, y_train_reg)
    models_dict['xgboost_regressor'] = xgb_reg
    
    # 4. Logistic Regression
    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_clf, y_train_clf)
    models_dict['logistic_regression'] = logreg
    
    # 5. Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_clf, y_train_clf)
    models_dict['random_forest_classifier'] = rf_clf
    
    # 6. XGBoost Classifier
    print("Training XGBoost Classifier...")
    xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train_clf, y_train_clf)
    models_dict['xgboost_classifier'] = xgb_clf
    
    # Store preprocessing objects
    models_dict['scaler'] = scaler
    models_dict['minmax_scaler'] = minmax_scaler
    models_dict['variance_selector'] = selector
    models_dict['feature_columns'] = feature_cols
    models_dict['selected_features'] = selected_features
    models_dict['median_rent'] = df_processed['monthly_rent'].median()
    
    # Save models to pickle files
    models_dir = 'backend/models'
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nSaving models to {models_dir}/...")
    
    # Save individual models
    for model_name, model_obj in models_dict.items():
        if model_name not in ['feature_columns', 'selected_features', 'median_rent']:
            with open(f'{models_dir}/{model_name}.pkl', 'wb') as f:
                pickle.dump(model_obj, f)
            print(f"Saved {model_name}.pkl")
    
    # Save a complete package with all models and metadata
    with open(f'{models_dir}/house_rental_models.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    print("Saved house_rental_models.pkl (complete package)")
    
    # Save feature metadata separately for easy access
    feature_metadata = {
        'feature_columns': feature_cols,
        'selected_features': selected_features,
        'median_rent': models_dict['median_rent']
    }
    
    with open(f'{models_dir}/feature_metadata.pkl', 'wb') as f:
        pickle.dump(feature_metadata, f)
    print("Saved feature_metadata.pkl")
    
    print(f"\nModel training and serialization completed!")
    print(f"All models saved in: {models_dir}/")
    
    return models_dict

if __name__ == "__main__":
    models = train_and_save_models()
    print("\n✅ Model extraction and serialization completed successfully!")