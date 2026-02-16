# A-list housings - Prediction System

A statistically robust system that predicts house prices 5 years into the future by decomposing the prediction into market-level growth and house-specific features.

## Overview

This system uses two complementary models:

1. **SARIMA Time Series Model**: Forecasts Canada's New Housing Price Index (HPI) to capture market-level growth
2. **Hedonic Pricing Model**: Uses XGBoost to model how house features affect relative pricing

**Final Formula**: `future_price = current_price × market_growth × house_multiplier`

## Statistical Justification

This decomposition is valid because:

- **Separability**: Market trends (HPI) and house characteristics operate through different mechanisms
- **Time Invariance**: House features don't change, but their market value does
- **Multiplicative Model**: Percentage changes compound, aligning with real estate market behavior
- **Cross-sectional vs Time-series**: We avoid treating house listings as a time series (they're a snapshot)
- **Hedonic Theory**: Well-established economic framework for decomposing goods into characteristics

## Installation

```bash
# Clone repository
cd cxc-hackthon

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv add pandas numpy statsmodels scikit-learn xgboost matplotlib seaborn pmdarima
```

## Usage

### Train the System

```bash
python house_price_predictor.py
```

This will:
- Train SARIMA model on HPI data (1981-2025)
- Train XGBoost hedonic pricing model on 44,000+ housing listings
- Generate forecast visualization (`hpi_forecast.png`)
- Save trained models to `models/predictor.pkl`
- Display example predictions

### Make Predictions

Using the command-line interface:

```bash
# Basic prediction
python predict.py --price 600000 --province BC --bedrooms 3 --bathrooms 2 --sqft 1800

# Detailed prediction with all features
python predict.py \
  --price 850000 \
  --province AB \
  --property-type "Single Family" \
  --bedrooms 4 \
  --bathrooms 3 \
  --sqft 2500 \
  --acreage 0.15 \
  --garage Yes \
  --parking Yes \
  --basement Finished \
  --fireplace Yes \
  --heating "forced air" \
  --pool No \
  --waterfront No
```

### Programmatic Usage

```python
from house_price_predictor import HousePricePredictor

# Initialize and load trained models
predictor = HousePricePredictor()
predictor.load_models('models/predictor.pkl')

# Define house features
house_features = {
    'Province': 'BC',
    'Property Type': 'Single Family',
    'Bedrooms': 3.0,
    'Bathrooms': 2.0,
    'Square Footage': 1800.0,
    'Acreage': 0.12,
    'Garage': 'Yes',
    'Parking': 'Yes',
    'Basement': 'Finished',
    'Fireplace': 'Yes',
    'Heating': 'forced air',
    'Pool': 'No',
    'Waterfront': 'No'
}

# Predict future price
future_price, breakdown = predictor.predict_future_price(
    current_price=650000,
    house_features=house_features
)

# Display results
predictor.print_prediction(breakdown)

print(f"Predicted price in 5 years: ${future_price:,.0f}")
print(f"Total appreciation: {breakdown['total_appreciation']:.2f}%")
```

## Datasets

### Dataset 1: `18100205.csv`
- **Source**: Statistics Canada New Housing Price Index (HPI)
- **Role**: Models market-level price growth over time
- **Period**: January 1981 - December 2025 (540 monthly observations)
- **Model**: SARIMA(1,1,1)(1,1,1,12) time series model
- **Output**: Market growth factor over 5 years

### Dataset 2: `cleaned_canada.csv`
- **Source**: Cross-sectional snapshot of Canadian housing listings (Feb 16, 2025)
- **Role**: Models how house features affect price relative to market
- **Records**: 44,745 housing listings across Canada
- **Model**: XGBoost log-linear regression
- **Output**: House-specific multiplier

## Model Performance

### SARIMA (Market Growth)
- **AIC**: -48.92
- **BIC**: -27.72
- **5-Year Market Growth**: +5.12%

### Hedonic Pricing Model
- **Train R²**: 0.818
- **Test R²**: 0.790
- **Test MAE**: $306,806

### Top Features by Importance
1. Province (32.4%)
2. Square Footage (21.7%)
3. Bathrooms (16.6%)
4. Basement (5.8%)
5. Garage (5.4%)

## Output Structure

```
================================================================================
5-YEAR PRICE PREDICTION
================================================================================

Current Price:              $650,000
Expected Current Price:     $698,432

--- Growth Components ---
Market Growth Factor:       1.0512 (+5.12%)
House Feature Adjustment:   1.0745 (+7.45%)

--- Final Prediction ---
Predicted Future Price:     $735,623
Total Appreciation:         +13.17%
Dollar Gain:                $85,623
================================================================================
```

## Project Structure

```
cxc-hackthon/
├── data/
│   ├── 18100205.csv              # Statistics Canada HPI data
│   └── cleaned_canada.csv        # Housing listings
├── models/
│   └── predictor.pkl             # Trained models
├── house_price_predictor.py      # Main system implementation
├── predict.py                    # CLI for predictions
├── hpi_forecast.png              # Forecast visualization
└── README_PREDICTION.md          # This file
```

## Key Constraints (Followed)

✅ **No fabricated timestamps**: Uses only actual HPI dates  
✅ **No per-house time series**: Houses treated as cross-sectional data  
✅ **Separated concerns**: Market growth and house features modeled independently  
✅ **Realistic predictions**: Based on established economic theory and real data

## Example Predictions

### Example 1: 3BR House in Calgary, AB
- **Current Price**: $500,000
- **Features**: 3 bed, 2 bath, 1500 sqft, finished basement
- **5-Year Prediction**: $516,596 (+3.32%)

### Example 2: 2BR Condo in Vancouver, BC
- **Current Price**: $750,000
- **Features**: 2 bed, 2 bath, 900 sqft, no basement
- **5-Year Prediction**: $639,634 (-14.72%)
- **Note**: Overvalued relative to market features

### Example 3: 5BR Luxury House in West Kelowna, BC
- **Current Price**: $1,200,000
- **Features**: 5 bed, 4 bath, 3500 sqft, finished basement
- **5-Year Prediction**: $1,469,396 (+22.45%)

## Visualization

The system generates `hpi_forecast.png` showing:
- Historical HPI from 1981-2025
- 5-year SARIMA forecast with confidence intervals
- Market growth trajectory

## API Reference

### `HousePricePredictor`

**Methods**:
- `train()`: Train both SARIMA and hedonic models
- `predict_future_price(current_price, house_features)`: Generate 5-year prediction
- `save_models(filepath)`: Save trained models
- `load_models(filepath)`: Load pre-trained models

**Prediction Output**:
```python
{
    'current_price': float,
    'market_growth_factor': float,
    'house_multiplier': float,
    'expected_current_price': float,
    'future_price': float,
    'total_appreciation': float (percentage),
    'market_appreciation': float (percentage),
    'feature_adjustment': float (percentage)
}
```

## Requirements

- Python 3.11+
- pandas
- numpy
- statsmodels
- scikit-learn
- xgboost
- matplotlib
- seaborn
- pmdarima

## Notes

- The system is trained on data up to December 2025
- Predictions assume market conditions follow historical patterns
- Individual predictions may vary due to local market factors not captured in national HPI
- The house_multiplier adjusts for relative valuation (overpriced/underpriced)

## License

Educational project for CXC Hackathon 2026

---

**Created**: February 2026  
**Model Version**: 1.0  
**Last Updated**: February 7, 2026
