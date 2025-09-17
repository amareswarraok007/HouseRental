"""
House Rental Flask API

A Flask web service for house rental price prediction and classification.
Provides REST API endpoints for machine learning model inference.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from typing import Dict, Any

# Import our model functions
from model import (
    load_models, 
    predict_monthly_rent, 
    classify_rent_category, 
    get_model_information,
    get_feature_importance,
    SAMPLE_INPUT
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models when the app starts
try:
    load_models()
    logger.info("✅ Models loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading models: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint - provides API information
    """
    return jsonify({
        'message': 'House Rental Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /models': 'Available models information',
            'POST /predict/rent': 'Predict monthly rent',
            'POST /predict/category': 'Classify rent as high/low',
            'GET /feature-importance/<model_type>': 'Get feature importance',
            'GET /sample-input': 'Get sample input format'
        },
        'status': 'active'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'API is running',
        'models_loaded': True
    })

@app.route('/models', methods=['GET'])
def models_info():
    """
    Get information about available models
    """
    try:
        info = get_model_information()
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/predict/rent', methods=['POST'])
def predict_rent():
    """
    Predict monthly rent using regression models
    
    Expected JSON payload:
    {
        "features": {
            "bed_count": 2,
            "bath_count": 1,
            "area_in_sqft": 1000,
            ...
        },
        "model_type": "random_forest_regressor"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing required field: features',
                'status': 'error'
            }), 400
        
        features = data['features']
        model_type = data.get('model_type', 'random_forest_regressor')
        
        # Validate model type
        valid_regression_models = ['linear_regression', 'random_forest_regressor', 'xgboost_regressor']
        if model_type not in valid_regression_models:
            return jsonify({
                'error': f'Invalid model type. Choose from: {valid_regression_models}',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = predict_monthly_rent(features, model_type)
        
        if result.get('status') == 'error':
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_rent: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'status': 'error'
        }), 500

@app.route('/predict/category', methods=['POST'])
def predict_category():
    """
    Classify rent as high or low using classification models
    
    Expected JSON payload:
    {
        "features": {
            "bed_count": 2,
            "bath_count": 1,
            "area_in_sqft": 1000,
            ...
        },
        "model_type": "random_forest_classifier"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing required field: features',
                'status': 'error'
            }), 400
        
        features = data['features']
        model_type = data.get('model_type', 'random_forest_classifier')
        
        # Validate model type
        valid_classification_models = ['logistic_regression', 'random_forest_classifier', 'xgboost_classifier']
        if model_type not in valid_classification_models:
            return jsonify({
                'error': f'Invalid model type. Choose from: {valid_classification_models}',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = classify_rent_category(features, model_type)
        
        if result.get('status') == 'error':
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_category: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'status': 'error'
        }), 500

@app.route('/feature-importance/<model_type>', methods=['GET'])
def feature_importance(model_type):
    """
    Get feature importance for tree-based models
    """
    try:
        result = get_feature_importance(model_type)
        
        if result.get('status') == 'error':
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in feature_importance: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'status': 'error'
        }), 500

@app.route('/sample-input', methods=['GET'])
def sample_input():
    """
    Get sample input format for API endpoints
    """
    return jsonify({
        'sample_input': SAMPLE_INPUT,
        'description': 'Use this format for the features field in prediction requests',
        'notes': [
            'Binary categorical features should be 0 or 1',
            'Numerical features should be provided as numbers',
            'Missing features will be set to default values'
        ]
    })

@app.route('/predict/both', methods=['POST'])
def predict_both():
    """
    Get both rent prediction and classification in a single request
    
    Expected JSON payload:
    {
        "features": {
            "bed_count": 2,
            "bath_count": 1,
            ...
        },
        "regression_model": "random_forest_regressor",  // optional
        "classification_model": "random_forest_classifier"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing required field: features',
                'status': 'error'
            }), 400
        
        features = data['features']
        regression_model = data.get('regression_model', 'random_forest_regressor')
        classification_model = data.get('classification_model', 'random_forest_classifier')
        
        # Get both predictions
        rent_prediction = predict_monthly_rent(features, regression_model)
        category_prediction = classify_rent_category(features, classification_model)
        
        result = {
            'rent_prediction': rent_prediction,
            'category_prediction': category_prediction,
            'input_features': features,
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_both: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation at the root endpoint',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on the server',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting House Rental API on port {port}")
    logger.info(f"🐛 Debug mode: {debug_mode}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )