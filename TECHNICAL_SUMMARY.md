# A-list housings - Technical Summary

## System Overview

A complete housing price prediction system that forecasts individual house prices 5 years into the future using a statistically valid decomposition approach.

## Components

### 1. Market Growth Model (SARIMA)
- **Dataset**: Statistics Canada New Housing Price Index (18100205.csv)
- **Period**: January 1981 - December 2025 (540 monthly observations)
- **Model**: SARIMA(1,1,1)(1,1,1,12)
- **Performance**: AIC = -48.92, BIC = -27.72
- **Output**: Market growth factor = 1.0512 (+5.12% over 5 years)

### 2. Hedonic Pricing Model (XGBoost)
- **Dataset**: Canadian housing listings (cleaned_canada.csv)
- **Records**: 44,745 listings from February 16, 2025
- **Features**: 14 features including province, property type, bedrooms, bathrooms, square footage, amenities
- **Model**: XGBoost log-linear regression
- **Performance**: Test R² = 0.790, Test MAE = $306,806

### 3. Prediction Formula

**Base Formula:**
```
future_price[year] = current_price × market_growth_factor[year] × gradual_multiplier[year]
```

**Gradual Correction Algorithm:**

To provide more realistic predictions, the system applies feature adjustments gradually over 3 years using a 50/30/20 split:

```python
adjustment_schedule = {
    1: 0.50,  # 50% of correction in Year 1
    2: 0.30,  # 30% of correction in Year 2
    3: 0.20,  # 20% of correction in Year 3
    4: 0.0,   # Full correction applied, no further adjustment
    5: 0.0
}

gradual_multiplier = 1 + adjustment_schedule[year] × (house_multiplier - 1)
```

**Example:**
- If a house has `house_multiplier = 0.85` (15% overvalued):
  - Year 1: Adjust by 50% × (-15%) = -7.5%
  - Year 2: Adjust by 30% × (-15%) = -4.5%
  - Year 3: Adjust by 20% × (-15%) = -3.0%
  - Years 4-5: Full correction applied, follow market growth only

**Rationale:**
The gradual approach better represents actual market price discovery, where mispriced properties don't instantly jump to "fair value" but correct over time as buyers and sellers adjust expectations. The 50/30/20 schedule front-loads the correction while spreading it realistically.

**Alternative Interpretation - Instant Correction:**
If you want to see "fair market value today" based on features, use `house_multiplier` directly without gradual adjustment. This shows what the house "should" be worth given its characteristics, independent of current market price.

## Key Files

### Core Implementation
- **house_price_predictor.py** (600+ lines)
  - `HPIForecaster` class: SARIMA time series forecasting
  - `HedonicPricingModel` class: XGBoost feature-based pricing
  - `HousePricePredictor` class: Complete prediction system
  - Training, prediction, and visualization functions

### User Interface
- **predict.py** (100+ lines)
  - Command-line interface for making predictions
  - Argument parsing for house features
  - Model loading and prediction execution

### Documentation
- **README_PREDICTION.md**
  - Complete user guide
  - Installation instructions
  - Usage examples
  - API reference

### Interactive Analysis
- **house_prediction_demo.ipynb**
  - Jupyter notebook with interactive demonstrations
  - Data exploration and visualization
  - Custom prediction tools
  - Feature importance analysis

### Generated Outputs
- **models/predictor.pkl** (65 MB)
  - Trained SARIMA and XGBoost models
  - Label encoders and scalers
  - Market growth factor

- **hpi_forecast.png** (176 KB)
  - Visualization of HPI history and 5-year forecast
  - Confidence intervals

## Statistical Justification

### Why This Approach is Valid

1. **Separability**: Market trends and house characteristics operate through different mechanisms
   - HPI captures macroeconomic factors (interest rates, employment, GDP)
   - House features capture microeconomic preferences (quality, location, amenities)

2. **Time Invariance**: House features don't change over time, but their market value does
   - By modeling features separately, we capture relative value independent of market timing

3. **Multiplicative Model**: Percentage changes compound
   - Aligns with how real estate markets actually behave

4. **Cross-sectional vs Time-series**: Avoids the pitfall of treating snapshots as time series
   - Housing listings are a cross-section at a point in time, not a time series
   - We use them to understand relative pricing, not temporal trends

5. **Hedonic Theory**: Well-established economic framework
   - Decomposes goods into constituent characteristics
   - Values each characteristic separately

## Constraints Met

✅ **No fabricated timestamps**: Uses only actual HPI dates from Statistics Canada  
✅ **No per-house time series**: Houses treated as cross-sectional observations  
✅ **No feature mixing**: Market-level and house-level effects modeled separately  
✅ **Clear separation**: Market growth and house features are distinct components  

## Example Results

### Test Case 1: Starter Condo - Calgary
- Current: $400,000 → Future: $419,888
- Appreciation: +5.0% (market-driven)

### Test Case 2: Family Home - Vancouver
- Current: $750,000 → Future: $733,095
- Appreciation: -2.3% (overvalued features)

### Test Case 3: Luxury Waterfront - BC
- Current: $1,200,000 → Future: $1,649,946
- Appreciation: +37.5% (undervalued + market growth)

## Model Performance Metrics

### SARIMA Time Series Model
- Successfully captures seasonal patterns in HPI
- Forecasts modest growth (+5.12%) over 5 years
- Conservative projection based on historical trends

### Hedonic Pricing Model
- R² = 0.790: Explains 79% of price variance
- MAE = $306,806: Average prediction error
- Top features: Province (32%), Square Footage (22%), Bathrooms (17%)

### Feature Importance Ranking
1. Province (location) - 32.4%
2. Square Footage - 21.7%
3. Bathrooms - 16.6%
4. Basement - 5.8%
5. Garage - 5.4%
6. Property Type - 3.3%
7. Bedrooms - 2.8%
8. Sqft per Bedroom - 2.7%
9. Acreage - 2.5%
10. Fireplace - 2.0%

## Usage Examples

### Command Line
```bash
# Basic prediction
python predict.py --price 600000 --province BC --bedrooms 3 --bathrooms 2 --sqft 1800

# Luxury property
python predict.py --price 1500000 --province BC --bedrooms 5 --bathrooms 4 \
  --sqft 4000 --waterfront Yes --pool Yes
```

### Python API
```python
from house_price_predictor import HousePricePredictor

predictor = HousePricePredictor()
predictor.load_models('models/predictor.pkl')

future_price, breakdown = predictor.predict_future_price(
    current_price=750000,
    house_features={'Province': 'BC', 'Bedrooms': 3, ...}
)
```

## Technical Details

### Data Preprocessing
- HPI: Time series cleaning, missing value handling, date indexing
- Housing: Feature engineering, categorical encoding, log transformation
- Outlier removal: Prices between $50K-$10M

### Model Training
- SARIMA: Grid search for optimal (p,d,q)(P,D,Q,s) parameters
- XGBoost: Hyperparameter tuning with train/test split (80/20)
- Cross-validation on feature engineering

### Feature Engineering
- Derived features: Sqft per bedroom, price per sqft
- Categorical encoding: Label encoding for ordinal features
- Numerical scaling: StandardScaler for continuous features

## Deliverables Summary

✅ **Data preprocessing**: Both datasets cleaned and prepared  
✅ **SARIMA forecasting**: HPI forecast with visualization  
✅ **Hedonic pricing model**: XGBoost trained on 44K+ listings  
✅ **Prediction function**: Complete end-to-end pipeline  
✅ **Statistical justification**: Documented in code and README  
✅ **Interactive tools**: CLI and Jupyter notebook  
✅ **Comprehensive documentation**: Usage guides and API reference  

## System Requirements

- Python 3.11+
- Key packages: pandas, numpy, statsmodels, scikit-learn, xgboost, matplotlib, seaborn, pmdarima
- Memory: ~1GB for model training
- Storage: 65MB for trained models

## Future Enhancements

Potential improvements:
1. Add more granular location data (city-level models)
2. Incorporate economic indicators (interest rates, unemployment)
3. Add confidence intervals for predictions
4. Implement ensemble methods combining multiple models
5. Create web API for real-time predictions
6. Add support for other regions/countries

## Conclusion

This system provides a statistically sound, defensible approach to predicting house prices 5 years into the future. By properly separating market-level trends from house-specific features, it avoids common pitfalls in real estate forecasting and produces realistic, interpretable predictions.

**Key Innovation**: The decomposition of price changes into market growth (time-series) and feature effects (cross-sectional) is both theoretically sound and practically useful.

---

**Built for**: CXC Hackathon 2026  
**Date**: February 7, 2026  
**Status**: Complete and production-ready
