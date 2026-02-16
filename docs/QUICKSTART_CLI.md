# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Activate Environment
```bash
source .venv/bin/activate
```

### Step 2: Make a Prediction
```bash
# Predict price for a house
python predict.py \
  --price 750000 \
  --province BC \
  --bedrooms 3 \
  --bathrooms 2 \
  --sqft 2000 \
  --property-type "Single Family"
```

### Step 3: Explore Interactively
```bash
# Open the Jupyter notebook
jupyter notebook house_prediction_demo.ipynb
```

## 📊 What You Get

```
================================================================================
PRICE PREDICTIONS OVER 5 YEARS
================================================================================

Current Price:              $750,000
Expected Current Price:     $697,385

--- Growth Components ---
Market Growth (5-year):     1.0512 (+5.12%)
House Feature Adjustment:   0.9298 (-7.02%)

--- Year-by-Year Predictions ---
Year 1: $696,228  |  -7.17% total  |  -0.17% market  |  $-53,772
Year 2: $702,801  |  -6.29% total  |  +0.78% market  |  $-47,199
Year 3: $712,164  |  -5.04% total  |  +2.12% market  |  $-37,836
Year 4: $722,471  |  -3.67% total  |  +3.60% market  |  $-27,529
Year 5: $733,095  |  -2.25% total  |  +5.12% market  |  $-16,905

--- 5-Year Summary ---
Final Predicted Price:      $733,095
Total Appreciation:         -2.25%
Total Dollar Gain:          $-16,905
================================================================================
```

## 🔧 More Options

### Train Models from Scratch
```bash
python house_price_predictor.py
```

### Predict with All Features + Plot
```bash
python predict.py \
  --price 1200000 \
  --province BC \
  --property-type "Single Family" \
  --bedrooms 4 \
  --bathrooms 3 \
  --sqft 3000 \
  --acreage 0.25 \
  --garage Yes \
  --parking Yes \
  --basement Finished \
  --fireplace Yes \
  --heating "forced air" \
  --pool Yes \
  --waterfront Yes \
  --plot
```
This will generate a `price_trajectory.png` chart showing the year-by-year price growth.

### Use Python API
```python
from house_price_predictor import HousePricePredictor

# Load trained models
predictor = HousePricePredictor()
predictor.load_models('models/predictor.pkl')

# Define house
house = {
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

# Predict
future_price, breakdown = predictor.predict_future_price(650000, house)
predictor.print_prediction(breakdown)
```

## 📚 Documentation

- **README_PREDICTION.md** - Complete user guide
- **TECHNICAL_SUMMARY.md** - Technical details and justification
- **house_prediction_demo.ipynb** - Interactive demonstrations

## ✅ System Status

- ✅ Models trained and saved (65 MB)
- ✅ HPI forecast generated (hpi_forecast.png)
- ✅ CLI interface ready
- ✅ Jupyter notebook available
- ✅ Documentation complete

## 🎯 Key Features

1. **Market Growth Model**: SARIMA on Statistics Canada HPI
   - Year 1: -0.17% | Year 2: +0.78% | Year 3: +2.12%
   - Year 4: +3.60% | Year 5: +5.12%
2. **House Feature Model**: XGBoost on 44,745 listings (R² = 0.79)
3. **Year-by-Year Predictions**: See how your house value evolves each year
4. **Visual Charts**: Generate price trajectory plots with `--plot`

## 💡 Understanding Results

- **Market Growth Factor**: How much the overall market will grow
- **House Multiplier**: Adjustment based on your house features
  - `> 1.0` = House is undervalued, will appreciate faster
  - `< 1.0` = House is overvalued, will appreciate slower
  - `= 1.0` = House is fairly valued

## 🆘 Need Help?

- See [README_PREDICTION.md](README_PREDICTION.md) for detailed usage
- See [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) for technical details
- Open [house_prediction_demo.ipynb](house_prediction_demo.ipynb) for interactive guide

---

**Ready to use!** All models are trained and saved. Just run predictions!
