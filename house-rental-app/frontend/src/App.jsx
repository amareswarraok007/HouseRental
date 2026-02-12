import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    bed_count: 2,
    bath_count: 1,
    area_in_sqft: 1000,
    geo_lat: 40.7128,
    geo_lon: -74.0060,
    listing_category_House: 1,
    listing_category_Condo: 0,
    payment_currency_USD: 1,
    service_fee_applicable_True: 0,
    image_available_True: 1,
    payment_schedule_Weekly: 0,
    data_provider_Provider2: 0,
    pet_policy_Conditional: 0,
    'pet_policy_Not Allowed': 0,
    city_Houston: 0,
    'city_Los Angeles': 0,
    'city_New York': 1,
    region_code_IL: 0,
    region_code_NY: 1,
    region_code_TX: 0
  });

  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (e.target.checked ? 1 : 0) : 
              type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleCityChange = (selectedCity) => {
    const cityFeatures = {
      'city_Houston': 0,
      'city_Los Angeles': 0,
      'city_New York': 0
    };
    
    const regionFeatures = {
      'region_code_IL': 0,
      'region_code_NY': 0,
      'region_code_TX': 0
    };

    if (selectedCity === 'Houston') {
      cityFeatures.city_Houston = 1;
      regionFeatures.region_code_TX = 1;
    } else if (selectedCity === 'Los Angeles') {
      cityFeatures['city_Los Angeles'] = 1;
      // CA not in our features, will default to 0
    } else if (selectedCity === 'New York') {
      cityFeatures['city_New York'] = 1;
      regionFeatures.region_code_NY = 1;
    } else if (selectedCity === 'Chicago') {
      // Chicago is not in features, will be "Other"
      regionFeatures.region_code_IL = 1;
    }

    setFormData(prev => ({
      ...prev,
      ...cityFeatures,
      ...regionFeatures
    }));
  };

  const handleListingTypeChange = (selectedType) => {
    const typeFeatures = {
      'listing_category_House': 0,
      'listing_category_Condo': 0
    };

    if (selectedType === 'House') {
      typeFeatures.listing_category_House = 1;
    } else if (selectedType === 'Condo') {
      typeFeatures.listing_category_Condo = 1;
    }
    // Apartment is the baseline (both 0)

    setFormData(prev => ({
      ...prev,
      ...typeFeatures
    }));
  };

  const predictRent = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/predict/both', {
        features: formData
      });
      
      setPredictions(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getCurrentCity = () => {
    if (formData['city_New York']) return 'New York';
    if (formData.city_Houston) return 'Houston';
    if (formData['city_Los Angeles']) return 'Los Angeles';
    if (formData.region_code_IL) return 'Chicago';
    return 'Other';
  };

  const getCurrentListingType = () => {
    if (formData.listing_category_House) return 'House';
    if (formData.listing_category_Condo) return 'Condo';
    return 'Apartment';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏠 House Rental Price Predictor</h1>
        <p>Get AI-powered predictions for house rental prices</p>
      </header>

      <main className="main-content">
        <div className="form-container">
          <h2>Property Details</h2>
          
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="bed_count">Bedrooms:</label>
              <input
                type="number"
                id="bed_count"
                name="bed_count"
                value={formData.bed_count}
                onChange={handleChange}
                min="1"
                max="10"
              />
            </div>

            <div className="form-group">
              <label htmlFor="bath_count">Bathrooms:</label>
              <input
                type="number"
                id="bath_count"
                name="bath_count"
                value={formData.bath_count}
                onChange={handleChange}
                min="1"
                max="10"
                step="0.5"
              />
            </div>

            <div className="form-group">
              <label htmlFor="area_in_sqft">Area (sq ft):</label>
              <input
                type="number"
                id="area_in_sqft"
                name="area_in_sqft"
                value={formData.area_in_sqft}
                onChange={handleChange}
                min="200"
                max="10000"
              />
            </div>

            <div className="form-group">
              <label htmlFor="city">City:</label>
              <select
                id="city"
                value={getCurrentCity()}
                onChange={(e) => handleCityChange(e.target.value)}
              >
                <option value="New York">New York</option>
                <option value="Houston">Houston</option>
                <option value="Los Angeles">Los Angeles</option>
                <option value="Chicago">Chicago</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="listing_type">Property Type:</label>
              <select
                id="listing_type"
                value={getCurrentListingType()}
                onChange={(e) => handleListingTypeChange(e.target.value)}
              >
                <option value="Apartment">Apartment</option>
                <option value="House">House</option>
                <option value="Condo">Condo</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="geo_lat">Latitude:</label>
              <input
                type="number"
                id="geo_lat"
                name="geo_lat"
                value={formData.geo_lat}
                onChange={handleChange}
                step="0.0001"
                min="25"
                max="48"
              />
            </div>

            <div className="form-group">
              <label htmlFor="geo_lon">Longitude:</label>
              <input
                type="number"
                id="geo_lon"
                name="geo_lon"
                value={formData.geo_lon}
                onChange={handleChange}
                step="0.0001"
                min="-125"
                max="-70"
              />
            </div>
          </div>

          <div className="checkbox-group">
            <h3>Additional Features</h3>
            
            <label className="checkbox-label">
              <input
                type="checkbox"
                name="image_available_True"
                checked={formData.image_available_True === 1}
                onChange={handleChange}
              />
              Images Available
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                name="service_fee_applicable_True"
                checked={formData.service_fee_applicable_True === 1}
                onChange={handleChange}
              />
              Service Fee Applicable
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                name="payment_schedule_Weekly"
                checked={formData.payment_schedule_Weekly === 1}
                onChange={handleChange}
              />
              Weekly Payment Schedule
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                name="pet_policy_Conditional"
                checked={formData.pet_policy_Conditional === 1}
                onChange={handleChange}
              />
              Conditional Pet Policy
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                name="pet_policy_Not Allowed"
                checked={formData['pet_policy_Not Allowed'] === 1}
                onChange={handleChange}
              />
              No Pets Allowed
            </label>
          </div>

          <button 
            className="predict-button"
            onClick={predictRent}
            disabled={loading}
          >
            {loading ? 'Predicting...' : 'Predict Rent'}
          </button>
        </div>

        {error && (
          <div className="error-container">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}

        {predictions && (
          <div className="results-container">
            <h2>Prediction Results</h2>
            
            <div className="results-grid">
              <div className="result-card rent-prediction">
                <h3>💰 Predicted Rent</h3>
                <div className="predicted-amount">
                  ${predictions.rent_prediction.predicted_rent?.toLocaleString()}
                </div>
                <p className="model-info">
                  Model: {predictions.rent_prediction.model_used}
                </p>
              </div>

              <div className="result-card category-prediction">
                <h3>📊 Rent Category</h3>
                <div className={`predicted-category ${predictions.category_prediction.predicted_class === 'High Rent' ? 'high-rent' : 'low-rent'}`}>
                  {predictions.category_prediction.predicted_class}
                </div>
                <p className="confidence">
                  Confidence: {predictions.category_prediction.confidence}%
                </p>
                <p className="threshold">
                  Threshold: ${predictions.category_prediction.median_rent_threshold?.toFixed(0)}
                </p>
                
                <div className="probability-bars">
                  <div className="prob-bar">
                    <span>Low Rent:</span>
                    <div className="bar">
                      <div 
                        className="fill low-rent-fill" 
                        style={{width: `${predictions.category_prediction.probabilities.low_rent}%`}}
                      ></div>
                    </div>
                    <span>{predictions.category_prediction.probabilities.low_rent}%</span>
                  </div>
                  <div className="prob-bar">
                    <span>High Rent:</span>
                    <div className="bar">
                      <div 
                        className="fill high-rent-fill" 
                        style={{width: `${predictions.category_prediction.probabilities.high_rent}%`}}
                      ></div>
                    </div>
                    <span>{predictions.category_prediction.probabilities.high_rent}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React and Flask • Powered by Machine Learning</p>
      </footer>
    </div>
  );
}

export default App;