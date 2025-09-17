"""
House Rental Model Prediction Helper

This module provides functions to load trained models and make predictions
for house rental price estimation and high/low rent classification.
"""

import pickle
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Union

class HouseRentalPredictor:
    """
    A class to handle house rental predictions using pre-trained models
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the predictor with models directory
        
        Args:
            models_dir (str): Directory containing the model files
        """
        self.models_dir = models_dir
        self.models = {}
        self.feature_metadata = {}
        self.is_loaded = False
        
    def load_models(self):
        """
        Load all models and preprocessing objects from pickle files
        """
        try:
            # Load the complete model package
            with open(os.path.join(self.models_dir, 'house_rental_models.pkl'), 'rb') as f:
                self.models = pickle.load(f)
            
            # Load feature metadata
            with open(os.path.join(self.models_dir, 'feature_metadata.pkl'), 'rb') as f:
                self.feature_metadata = pickle.load(f)
            
            self.is_loaded = True
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available models
        
        Returns:
            Dict containing model information
        """
        if not self.is_loaded:
            self.load_models()
        
        model_info = {
            'regression_models': ['linear_regression', 'random_forest_regressor', 'xgboost_regressor'],
            'classification_models': ['logistic_regression', 'random_forest_classifier', 'xgboost_classifier'],
            'feature_count': len(self.feature_metadata['selected_features']),
            'median_rent': self.feature_metadata['median_rent']
        }
        
        return model_info
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data to match model expectations
        
        Args:
            input_data (dict): Dictionary containing input features
            
        Returns:
            np.ndarray: Preprocessed feature array
        """
        if not self.is_loaded:
            self.load_models()
        
        # Get the original features that the preprocessing pipeline expects
        original_features = self.feature_metadata['feature_columns']
        
        # Initialize all features with default values in the same order as training
        feature_dict = {}
        
        for feature in original_features:
            if feature in input_data:
                feature_dict[feature] = input_data[feature]
            else:
                # Set intelligent defaults based on feature name
                if feature in ['bed_count', 'bath_count']:
                    feature_dict[feature] = input_data.get(feature, 2 if 'bed' in feature else 1)
                elif feature == 'area_in_sqft':
                    feature_dict[feature] = input_data.get(feature, 1000)
                elif feature == 'geo_lat':
                    feature_dict[feature] = input_data.get(feature, 40.7)  # Default to NYC
                elif feature == 'geo_lon':
                    feature_dict[feature] = input_data.get(feature, -74.0)  # Default to NYC
                else:
                    # Binary categorical features - set intelligent defaults
                    if feature == 'listing_category_House':
                        feature_dict[feature] = 1  # Default to House
                    elif feature == 'payment_currency_USD':
                        feature_dict[feature] = 1  # Default to USD
                    elif feature == 'image_available_True':
                        feature_dict[feature] = 1  # Default to having images
                    elif feature == 'city_New York':
                        feature_dict[feature] = 1  # Default to NYC
                    elif feature == 'region_code_NY':
                        feature_dict[feature] = 1  # Default to NY
                    else:
                        feature_dict[feature] = 0  # Default binary features to 0
        
        # Convert to DataFrame with the original feature order
        df = pd.DataFrame([feature_dict], columns=original_features)
        
        # Apply scaling to numerical columns in the correct order expected by the scalers
        # The scaler expects: ['bath_count', 'bed_count', 'area_in_sqft', 'geo_lat', 'geo_lon']
        scaler_feature_order = ['bath_count', 'bed_count', 'area_in_sqft', 'geo_lat', 'geo_lon']
        
        # Create a copy for scaling
        df_scaled = df.copy()
        
        # Apply scaling to numerical columns in the correct order
        numerical_data = df[scaler_feature_order].values
        scaled_numerical = self.models['scaler'].transform(numerical_data)
        scaled_numerical = self.models['minmax_scaler'].transform(scaled_numerical)
        
        # Update the DataFrame with scaled values
        df_scaled[scaler_feature_order] = scaled_numerical
        
        # Apply variance threshold selection 
        df_selected = self.models['variance_selector'].transform(df_scaled)
        
        return df_selected
    
    def predict_rent(self, input_data: Dict[str, Any], model_type: str = 'random_forest_regressor') -> Dict[str, Any]:
        """
        Predict monthly rent using regression models
        
        Args:
            input_data (dict): Input features
            model_type (str): Type of regression model to use
            
        Returns:
            Dict containing prediction results
        """
        if not self.is_loaded:
            self.load_models()
        
        # Validate model type
        regression_models = ['linear_regression', 'random_forest_regressor', 'xgboost_regressor']
        if model_type not in regression_models:
            raise ValueError(f"Invalid model type. Choose from: {regression_models}")
        
        try:
            # Preprocess input
            processed_input = self.preprocess_input(input_data)
            
            # Make prediction
            model = self.models[model_type]
            prediction = model.predict(processed_input)[0]
            
            # Ensure prediction is positive
            prediction = max(prediction, 100)  # Minimum rent of $100
            
            result = {
                'predicted_rent': round(prediction, 2),
                'model_used': model_type,
                'input_features': input_data,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def classify_rent(self, input_data: Dict[str, Any], model_type: str = 'random_forest_classifier') -> Dict[str, Any]:
        """
        Classify rent as high or low using classification models
        
        Args:
            input_data (dict): Input features
            model_type (str): Type of classification model to use
            
        Returns:
            Dict containing classification results
        """
        if not self.is_loaded:
            self.load_models()
        
        # Validate model type
        classification_models = ['logistic_regression', 'random_forest_classifier', 'xgboost_classifier']
        if model_type not in classification_models:
            raise ValueError(f"Invalid model type. Choose from: {classification_models}")
        
        try:
            # Preprocess input
            processed_input = self.preprocess_input(input_data)
            
            # Make prediction
            model = self.models[model_type]
            prediction = model.predict(processed_input)[0]
            prediction_proba = model.predict_proba(processed_input)[0]
            
            # Get class labels
            class_labels = ['Low Rent', 'High Rent']
            predicted_class = class_labels[prediction]
            
            result = {
                'predicted_class': predicted_class,
                'confidence': round(max(prediction_proba) * 100, 2),
                'probabilities': {
                    'low_rent': round(prediction_proba[0] * 100, 2),
                    'high_rent': round(prediction_proba[1] * 100, 2)
                },
                'model_used': model_type,
                'median_rent_threshold': self.feature_metadata['median_rent'],
                'input_features': input_data,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_feature_importance(self, model_type: str = 'random_forest_regressor') -> Dict[str, Any]:
        """
        Get feature importance from tree-based models
        
        Args:
            model_type (str): Type of model to get importance from
            
        Returns:
            Dict containing feature importance information
        """
        if not self.is_loaded:
            self.load_models()
        
        tree_models = ['random_forest_regressor', 'random_forest_classifier', 
                      'xgboost_regressor', 'xgboost_classifier']
        
        if model_type not in tree_models:
            return {'error': f'Feature importance only available for: {tree_models}'}
        
        try:
            model = self.models[model_type]
            importances = model.feature_importances_
            feature_names = self.feature_metadata['selected_features']
            
            # Create importance pairs and sort
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 10 features
            top_features = importance_pairs[:10]
            
            result = {
                'model': model_type,
                'top_features': [
                    {'feature': name, 'importance': round(importance, 4)}
                    for name, importance in top_features
                ],
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }

# Global predictor instance
predictor = HouseRentalPredictor()

def load_models():
    """Load models into the global predictor instance"""
    predictor.load_models()

def predict_monthly_rent(input_data: Dict[str, Any], model_type: str = 'random_forest_regressor') -> Dict[str, Any]:
    """
    Predict monthly rent (wrapper function)
    """
    return predictor.predict_rent(input_data, model_type)

def classify_rent_category(input_data: Dict[str, Any], model_type: str = 'random_forest_classifier') -> Dict[str, Any]:
    """
    Classify rent as high/low (wrapper function)
    """
    return predictor.classify_rent(input_data, model_type)

def get_model_information() -> Dict[str, Any]:
    """
    Get model information (wrapper function)
    """
    return predictor.get_model_info()

def get_feature_importance(model_type: str = 'random_forest_regressor') -> Dict[str, Any]:
    """
    Get feature importance (wrapper function)
    """
    return predictor.get_feature_importance(model_type)

# Sample input format for reference - matches the exact original feature columns
SAMPLE_INPUT = {
    "bed_count": 2,
    "bath_count": 1,
    "area_in_sqft": 1000,
    "geo_lat": 40.7128,
    "geo_lon": -74.0060,
    "listing_category_Condo": 0,
    "listing_category_House": 1,
    "payment_currency_USD": 1,
    "service_fee_applicable_True": 0,
    "image_available_True": 1,
    "payment_schedule_Weekly": 0,
    "data_provider_Provider2": 0,
    "pet_policy_Conditional": 0,
    "pet_policy_Not Allowed": 0,
    "city_Houston": 0,
    "city_Los Angeles": 0,
    "city_New York": 1,
    "region_code_IL": 0,
    "region_code_NY": 1,
    "region_code_TX": 0
}