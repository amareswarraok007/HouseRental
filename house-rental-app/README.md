# 🏠 House Rental Price Predictor

A full-stack machine learning application that predicts house rental prices using AI models trained on rental data. The application provides both price estimation and rent category classification (high/low rent).

## 🌟 Features

- **Price Prediction**: Get accurate rental price estimates using multiple ML models
- **Rent Classification**: Classify properties as high or low rent with confidence scores
- **Multiple Models**: Choose from Random Forest, XGBoost, and Linear/Logistic Regression
- **Interactive UI**: Modern React frontend with real-time predictions
- **REST API**: Flask backend with comprehensive API endpoints
- **Feature Importance**: Analyze which features matter most for predictions

## 🏗️ Architecture

```
house-rental-app/
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   ├── model.py            # ML model loading and prediction logic
│   ├── models/             # Serialized ML models (.pkl files)
│   └── requirements.txt    # Python dependencies
├── frontend/               # React web application
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   └── App.css        # Styling
│   ├── public/
│   └── package.json       # Node.js dependencies
├── .devcontainer/         # Development container configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd house-rental-app
   ```

2. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Start the Flask backend** (in one terminal)
   ```bash
   cd backend
   python app.py
   ```
   The API will be available at `http://localhost:5000`

2. **Start the React frontend** (in another terminal)
   ```bash
   cd frontend
   npm start
   ```
   The web app will be available at `http://localhost:3000`

## 🔧 Development with DevContainer

This project includes a DevContainer configuration for easy development:

1. Open the project in VS Code
2. Install the "Dev Containers" extension
3. Press `Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
4. The container will automatically install all dependencies

## 📡 API Endpoints

### Base URL: `http://localhost:5000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check |
| `/models` | GET | Information about available models |
| `/predict/rent` | POST | Predict monthly rent |
| `/predict/category` | POST | Classify rent as high/low |
| `/predict/both` | POST | Get both rent prediction and classification |
| `/feature-importance/<model_type>` | GET | Get feature importance |
| `/sample-input` | GET | Get sample input format |

### Example API Usage

**Predict Rent:**
```bash
curl -X POST http://localhost:5000/predict/rent \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "bed_count": 2,
      "bath_count": 1,
      "area_in_sqft": 1000,
      "city_New York": 1,
      "listing_category_House": 1
    }
  }'
```

**Response:**
```json
{
  "predicted_rent": 2850.75,
  "model_used": "random_forest_regressor",
  "status": "success"
}
```

## 🤖 Machine Learning Models

The application includes several pre-trained models:

### Regression Models (Price Prediction)
- **Random Forest Regressor** ⭐ (Recommended)
- **XGBoost Regressor**
- **Linear Regression**

### Classification Models (High/Low Rent)
- **Random Forest Classifier** ⭐ (Recommended)
- **XGBoost Classifier**
- **Logistic Regression**

### Model Performance
The models were trained on house rental data with the following preprocessing:
- Feature scaling (StandardScaler + MinMaxScaler)
- One-hot encoding for categorical variables
- Outlier removal using IQR method
- Feature selection using variance threshold

## 📊 Input Features

The model accepts the following features:

### Numerical Features
- `bed_count`: Number of bedrooms (1-10)
- `bath_count`: Number of bathrooms (1-10)
- `area_in_sqft`: Property area in square feet
- `geo_lat`: Latitude coordinate
- `geo_lon`: Longitude coordinate

### Categorical Features (One-hot encoded)
- **Property Type**: House, Apartment, Condo
- **City**: New York, Houston, Los Angeles, Chicago, Other
- **Region**: NY, TX, IL, Other
- **Payment**: USD currency, Weekly schedule
- **Features**: Images available, Service fees, Pet policies

## 🎨 Frontend Features

The React frontend provides:
- **Intuitive Form**: Easy-to-use property input form
- **Real-time Predictions**: Instant ML model predictions
- **Visual Results**: Colorful prediction cards with confidence scores
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: User-friendly error messages

## 🔄 Model Training Pipeline

The models were trained using the following pipeline:

1. **Data Loading**: Load house rental dataset
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Engineering**: Create categorical features and scaling
4. **Model Training**: Train multiple ML algorithms
5. **Model Serialization**: Save models as pickle files
6. **API Integration**: Load models for real-time predictions

## 📈 Extending the Application

### Adding New Models
1. Train your model using the same preprocessing pipeline
2. Save the model as a pickle file in `backend/models/`
3. Update `model.py` to load and use the new model
4. Add new endpoints in `app.py` if needed

### Adding New Features
1. Update the feature list in the training pipeline
2. Retrain and serialize the models
3. Update the frontend form to include new inputs
4. Update the API to handle new features

## 🛠️ Troubleshooting

### Common Issues

**Models not loading:**
- Ensure all pickle files are in `backend/models/`
- Check that scikit-learn and xgboost versions match training environment

**API connection errors:**
- Verify Flask server is running on port 5000
- Check CORS configuration in `app.py`

**Frontend build issues:**
- Delete `node_modules` and run `npm install` again
- Ensure Node.js version is 16+

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

Built with ❤️ using React, Flask, and scikit-learn