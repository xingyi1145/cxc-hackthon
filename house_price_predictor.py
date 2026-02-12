"""
A-list housings - Price Prediction System

This system predicts house prices 5 years into the future by decomposing the prediction into:
1. Market-level growth (SARIMA model on HPI time series)
2. House-specific features (Hedonic pricing model)

Statistical Justification:
- The New Housing Price Index (HPI) captures macroeconomic trends, market cycles, and
  aggregate supply/demand dynamics at the national level.
- The cross-sectional housing data captures how specific features (bedrooms, location,
  amenities) affect relative pricing within the market.
- By separating these effects, we avoid conflating time-variant market forces with
  time-invariant house characteristics, leading to more robust predictions.

Formula: future_price = current_price × market_growth × house_multiplier
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle

warnings.filterwarnings('ignore')

class HPIForecaster:
    """Forecasts market-level Housing Price Index using SARIMA"""
    
    def __init__(self):
        self.model = None
        self.hpi_data = None
        self.current_hpi = None
        self.future_hpi = None
        self.market_growth_factor = None
        
    def load_and_preprocess_hpi(self, filepath='data/18100205.csv'):
        """Load and clean Statistics Canada HPI data"""
        print("Loading HPI dataset...")
        df = pd.read_csv(filepath)
        
        # Filter for Canada-level "Total (house and land)" index
        df_canada = df[
            (df['GEO'] == 'Canada') & 
            (df['New housing price indexes'] == 'Total (house and land)')
        ].copy()
        
        # Convert date and value
        df_canada['REF_DATE'] = pd.to_datetime(df_canada['REF_DATE'])
        df_canada['VALUE'] = pd.to_numeric(df_canada['VALUE'], errors='coerce')
        
        # Remove missing values
        df_canada = df_canada.dropna(subset=['VALUE'])
        
        # Sort by date
        df_canada = df_canada.sort_values('REF_DATE')
        
        # Set date as index
        df_canada.set_index('REF_DATE', inplace=True)
        
        # Extract the HPI series
        self.hpi_data = df_canada['VALUE']
        
        print(f"Loaded {len(self.hpi_data)} monthly observations from {self.hpi_data.index[0]} to {self.hpi_data.index[-1]}")
        print(f"HPI range: {self.hpi_data.min():.2f} to {self.hpi_data.max():.2f}")
        
        return self.hpi_data
    
    def train_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Train SARIMA model on HPI time series"""
        print("\nTraining SARIMA model...")
        print(f"Order: {order}, Seasonal Order: {seasonal_order}")
        
        # Fit SARIMA model
        self.model = SARIMAX(
            self.hpi_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.model_fit = self.model.fit(disp=False, maxiter=200)
        
        print("SARIMA model trained successfully!")
        print(f"AIC: {self.model_fit.aic:.2f}")
        print(f"BIC: {self.model_fit.bic:.2f}")
        
        return self.model_fit
    
    def forecast_5_years(self):
        """Forecast HPI 5 years (60 months) into the future"""
        print("\nForecasting 5 years ahead...")
        
        # Get current HPI (most recent value)
        self.current_hpi = self.hpi_data.iloc[-1]
        current_date = self.hpi_data.index[-1]
        
        # Forecast 60 months ahead
        forecast = self.model_fit.forecast(steps=60)
        
        # Get HPI at 1, 2, 3, 4, and 5 years (12, 24, 36, 48, 60 months)
        self.yearly_hpi = {
            1: forecast.iloc[11],   # 12 months (index 11)
            2: forecast.iloc[23],   # 24 months (index 23)
            3: forecast.iloc[35],   # 36 months (index 35)
            4: forecast.iloc[47],   # 48 months (index 47)
            5: forecast.iloc[59]    # 60 months (index 59)
        }
        
        # Calculate market growth factors for each year
        self.yearly_growth_factors = {
            year: hpi / self.current_hpi 
            for year, hpi in self.yearly_hpi.items()
        }
        
        # Keep the 5-year values for backward compatibility
        self.future_hpi = self.yearly_hpi[5]
        self.market_growth_factor = self.yearly_growth_factors[5]
        
        print(f"Current HPI ({current_date.strftime('%Y-%m')}): {self.current_hpi:.2f}")
        print(f"\nYearly HPI Forecasts:")
        for year in range(1, 6):
            hpi = self.yearly_hpi[year]
            growth = self.yearly_growth_factors[year]
            print(f"  Year {year}: HPI = {hpi:.2f}, Growth = {growth:.4f} ({(growth-1)*100:.2f}%)")
        
        return self.market_growth_factor
    
    def plot_forecast(self, save_path='hpi_forecast.png'):
        """Plot historical HPI and 5-year forecast"""
        forecast = self.model_fit.forecast(steps=60)
        forecast_ci = self.model_fit.get_forecast(steps=60).conf_int()
        
        plt.figure(figsize=(14, 6))
        
        # Plot historical data
        plt.plot(self.hpi_data.index, self.hpi_data.values, label='Historical HPI', color='blue')
        
        # Plot forecast
        forecast_dates = pd.date_range(
            start=self.hpi_data.index[-1] + pd.DateOffset(months=1),
            periods=60,
            freq='MS'
        )
        plt.plot(forecast_dates, forecast.values, label='5-Year Forecast', color='red', linestyle='--')
        
        # Plot confidence interval
        plt.fill_between(
            forecast_dates,
            forecast_ci.iloc[:, 0],
            forecast_ci.iloc[:, 1],
            color='red',
            alpha=0.2
        )
        
        plt.axvline(self.hpi_data.index[-1], color='gray', linestyle=':', label='Current Date')
        plt.xlabel('Date')
        plt.ylabel('HPI (Index, 2016-12=100)')
        plt.title('Canadian Housing Price Index: Historical and 5-Year Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nForecast plot saved to {save_path}")
        plt.close()


class HedonicPricingModel:
    """Models house-specific price adjustments based on features"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_features = None
        self.numerical_features = None
        self.data = None
        
    def load_and_preprocess_housing(self, filepath='data/cleaned_canada.csv'):
        """Load and engineer features from housing listings"""
        print("\n" + "="*80)
        print("Loading housing listings dataset...")
        df = pd.read_csv(filepath)
        
        print(f"Loaded {len(df)} housing listings")
        print(f"Columns: {df.columns.tolist()}")
        
        # Store original data
        self.data = df.copy()
        
        # Remove rows with missing price
        df = df.dropna(subset=['Price'])
        
        # Remove unrealistic prices (likely errors)
        df = df[(df['Price'] >= 50000) & (df['Price'] <= 10000000)]
        
        print(f"After cleaning: {len(df)} listings")
        print(f"Price range: ${df['Price'].min():,.0f} to ${df['Price'].max():,.0f}")
        print(f"Median price: ${df['Price'].median():,.0f}")
        
        return df
    
    def engineer_features(self, df):
        """Create and select features for modeling"""
        print("\nEngineering features...")
        
        # Define categorical and numerical features
        self.categorical_features = [
            'Province', 'Property Type', 'Garage', 'Parking', 
            'Basement', 'Fireplace', 'Heating', 'Pool', 'Waterfront'
        ]
        
        self.numerical_features = [
            'Bedrooms', 'Bathrooms', 'Acreage', 'Square Footage'
        ]
        
        # Filter to available columns
        self.categorical_features = [f for f in self.categorical_features if f in df.columns]
        self.numerical_features = [f for f in self.numerical_features if f in df.columns]
        
        # Handle missing values in numerical features
        for col in self.numerical_features:
            df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values in categorical features
        for col in self.categorical_features:
            df[col] = df[col].fillna('Unknown')
        
        # Create derived features
        if 'Square Footage' in df.columns and 'Bedrooms' in df.columns:
            df['Sqft_per_Bedroom'] = df['Square Footage'] / (df['Bedrooms'] + 1)
        
        if 'Price' in df.columns and 'Square Footage' in df.columns:
            df['Price_per_Sqft'] = df['Price'] / (df['Square Footage'] + 1)
        
        # Encode categorical features
        df_encoded = df.copy()
        for col in self.categorical_features:
            self.label_encoders[col] = LabelEncoder()
            df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        
        # Prepare feature matrix
        feature_cols = self.numerical_features + [col + '_encoded' for col in self.categorical_features]
        
        # Add derived features if they exist
        if 'Sqft_per_Bedroom' in df_encoded.columns:
            feature_cols.append('Sqft_per_Bedroom')
        
        self.feature_columns = feature_cols
        
        X = df_encoded[feature_cols]
        y = np.log(df['Price'])  # Log transform for log-linear model
        
        print(f"Features used: {self.feature_columns}")
        print(f"Total features: {len(self.feature_columns)}")
        
        return X, y, df_encoded
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train XGBoost hedonic pricing model"""
        print("\n" + "="*80)
        print("Training hedonic pricing model...")
        
        # Engineer features
        X, y, df_encoded = self.engineer_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Convert back from log space for evaluation
        train_mae = mean_absolute_error(np.exp(y_train), np.exp(y_train_pred))
        test_mae = mean_absolute_error(np.exp(y_test), np.exp(y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\nModel Performance:")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Train MAE: ${train_mae:,.0f}")
        print(f"Test MAE: ${test_mae:,.0f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self.model
    
    def predict_house_multiplier(self, house_features):
        """
        Predict house-specific price multiplier relative to market average
        
        Args:
            house_features: dict with house characteristics
            
        Returns:
            multiplier: float, adjustment factor relative to market average
        """
        # Create feature vector
        feature_vector = {}
        
        # Numerical features
        for col in self.numerical_features:
            feature_vector[col] = house_features.get(col, self.data[col].median())
        
        # Categorical features
        for col in self.categorical_features:
            value = house_features.get(col, 'Unknown')
            if value in self.label_encoders[col].classes_:
                encoded_value = self.label_encoders[col].transform([value])[0]
            else:
                # Use most common class for unknown values
                encoded_value = 0
            feature_vector[col + '_encoded'] = encoded_value
        
        # Derived features
        if 'Sqft_per_Bedroom' in self.feature_columns:
            sqft = feature_vector.get('Square Footage', 1500)
            beds = feature_vector.get('Bedrooms', 3)
            feature_vector['Sqft_per_Bedroom'] = sqft / (beds + 1)
        
        # Create DataFrame
        X_pred = pd.DataFrame([feature_vector])[self.feature_columns]
        
        # Predict log price
        log_price_pred = self.model.predict(X_pred)[0]
        
        # Calculate expected price based on features
        expected_price = np.exp(log_price_pred)
        
        # Current price
        current_price = house_features.get('current_price', expected_price)
        
        # Multiplier: how much this house deviates from market average
        # If house is undervalued, multiplier > 1; if overvalued, multiplier < 1
        multiplier = expected_price / current_price
        
        return multiplier, expected_price


class HousePricePredictor:
    """Complete system for predicting house prices 5 years in the future"""
    
    def __init__(self):
        self.hpi_forecaster = HPIForecaster()
        self.hedonic_model = HedonicPricingModel()
        self.market_growth_factor = None
        
    def train(self, hpi_filepath='data/18100205.csv', housing_filepath='data/cleaned_canada.csv'):
        """Train both components of the prediction system"""
        print("="*80)
        print("A-LIST HOUSINGS - TRAINING")
        print("="*80)
        
        # 1. Train SARIMA on HPI
        print("\n[STEP 1] Training Market Growth Model (SARIMA on HPI)")
        print("-"*80)
        self.hpi_forecaster.load_and_preprocess_hpi(hpi_filepath)
        self.hpi_forecaster.train_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        self.market_growth_factor = self.hpi_forecaster.forecast_5_years()
        self.hpi_forecaster.plot_forecast('hpi_forecast.png')
        
        # 2. Train hedonic model on housing features
        print("\n[STEP 2] Training House Feature Model (Hedonic Pricing)")
        print("-"*80)
        df_housing = self.hedonic_model.load_and_preprocess_housing(housing_filepath)
        self.hedonic_model.train_model(df_housing)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\nMarket Growth Factor (5 years): {self.market_growth_factor:.4f}")
        print(f"Expected market appreciation: {(self.market_growth_factor-1)*100:.2f}%")
        
    def predict_future_price(self, current_price, house_features):
        """
        Predict house price for years 1 through 5
        
        Args:
            current_price: float, current house price
            house_features: dict, house characteristics (bedrooms, bathrooms, etc.)
            
        Returns:
            yearly_predictions: dict, predicted prices for years 1-5
            breakdown: dict, detailed breakdown of prediction
        """
        # Add current price to features
        house_features['current_price'] = current_price
        
        # Get house-specific multiplier
        house_multiplier, expected_price = self.hedonic_model.predict_house_multiplier(house_features)
        
        # Gradual adjustment: 50% Year 1, 30% Year 2, 20% Year 3, then full
        adjustment_schedule = {
            1: 0.50,  # 50% of adjustment in year 1
            2: 0.80,  # 80% cumulative by year 2 (50% + 30%)
            3: 1.00,  # 100% by year 3 (50% + 30% + 20%)
            4: 1.00,  # Full adjustment
            5: 1.00   # Full adjustment
        }
        
        # Calculate future prices for each year with gradual correction
        yearly_predictions = {}
        for year in range(1, 6):
            market_growth = self.hpi_forecaster.yearly_growth_factors[year]
            # Apply gradual multiplier: 1 + (adjustment_pct × (house_multiplier - 1))
            gradual_multiplier = 1 + adjustment_schedule[year] * (house_multiplier - 1)
            yearly_predictions[year] = current_price * market_growth * gradual_multiplier
        
        # Breakdown (using 5-year values for summary)
        breakdown = {
            'current_price': current_price,
            'yearly_growth_factors': self.hpi_forecaster.yearly_growth_factors,
            'yearly_predictions': yearly_predictions,
            'house_multiplier': house_multiplier,
            'expected_current_price': expected_price,
            'final_price': yearly_predictions[5],
            'total_appreciation': (yearly_predictions[5] / current_price - 1) * 100,
            'market_appreciation': (self.market_growth_factor - 1) * 100,
            'feature_adjustment': (house_multiplier - 1) * 100
        }
        
        return yearly_predictions, breakdown
    
    def print_prediction(self, breakdown):
        """Pretty print prediction breakdown"""
        print("\n" + "="*80)
        print("PRICE PREDICTIONS OVER 5 YEARS")
        print("="*80)
        print(f"\nCurrent Price:              ${breakdown['current_price']:,.0f}")
        print(f"Expected Current Price:     ${breakdown['expected_current_price']:,.0f}")
        
        print(f"\n--- Growth Components ---")
        print(f"Market Growth (5-year):     {breakdown['yearly_growth_factors'][5]:.4f} ({breakdown['market_appreciation']:+.2f}%)")
        print(f"House Feature Adjustment:   {breakdown['house_multiplier']:.4f} ({breakdown['feature_adjustment']:+.2f}%)")
        
        print(f"\n--- Year-by-Year Predictions ---")
        yearly_preds = breakdown['yearly_predictions']
        for year in range(1, 6):
            price = yearly_preds[year]
            appreciation = (price / breakdown['current_price'] - 1) * 100
            dollar_gain = price - breakdown['current_price']
            market_growth_pct = (breakdown['yearly_growth_factors'][year] - 1) * 100
            print(f"Year {year}: ${price:,.0f}  |  {appreciation:+.2f}% total  |  {market_growth_pct:+.2f}% market  |  ${dollar_gain:+,.0f}")
        
        print(f"\n--- 5-Year Summary ---")
        print(f"Final Predicted Price:      ${breakdown['final_price']:,.0f}")
        print(f"Total Appreciation:         {breakdown['total_appreciation']:+.2f}%")
        print(f"Total Dollar Gain:          ${breakdown['final_price'] - breakdown['current_price']:+,.0f}")
        print("="*80)
    
    def plot_prediction(self, breakdown, save_path='price_trajectory.png'):
        """Plot the year-by-year price trajectory"""
        years = [0] + list(range(1, 6))
        prices = [breakdown['current_price']] + [breakdown['yearly_predictions'][y] for y in range(1, 6)]
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, prices, marker='o', linewidth=2, markersize=10, color='#2ecc71' if prices[-1] > prices[0] else '#e74c3c')
        
        # Add value labels
        for year, price in zip(years, prices):
            plt.annotate(f'${price:,.0f}', 
                        xy=(year, price), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        fontweight='bold')
        
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title('House Price Trajectory - 5 Year Forecast', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(years)
        
        # Add appreciation info
        total_change = breakdown['total_appreciation']
        color = '#2ecc71' if total_change > 0 else '#e74c3c'
        plt.text(0.02, 0.98, f"Total Change: {total_change:+.2f}%\nMarket: {breakdown['market_appreciation']:+.2f}%\nFeature Adj: {breakdown['feature_adjustment']:+.2f}%",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Price trajectory plot saved to {save_path}")
        plt.close()
    
    def save_models(self, filepath='models/predictor.pkl'):
        """Save trained models"""
        import os
        os.makedirs('models', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'hpi_forecaster': self.hpi_forecaster,
                'hedonic_model': self.hedonic_model,
                'market_growth_factor': self.market_growth_factor
            }, f)
        print(f"\nModels saved to {filepath}")
    
    def load_models(self, filepath='models/predictor.pkl'):
        """Load trained models (handles pickle saved from __main__)"""
        import io
        
        class _Unpickler(pickle.Unpickler):
            """Redirect __main__ references to house_price_predictor module"""
            def find_class(self, module, name):
                if module == '__main__':
                    module = 'house_price_predictor'
                return super().find_class(module, name)
        
        with open(filepath, 'rb') as f:
            data = _Unpickler(f).load()
        self.hpi_forecaster = data['hpi_forecaster']
        self.hedonic_model = data['hedonic_model']
        self.market_growth_factor = data['market_growth_factor']
        print(f"Models loaded from {filepath}")


if __name__ == "__main__":
    # Initialize and train the complete system
    predictor = HousePricePredictor()
    predictor.train(
        hpi_filepath='data/18100205.csv',
        housing_filepath='data/cleaned_canada.csv'
    )
    
    # Save models
    predictor.save_models('models/predictor.pkl')
    
    # Example predictions
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    # Example 1: 3-bedroom house in Calgary
    example1 = {
        'Province': 'AB',
        'Property Type': 'Single Family',
        'Bedrooms': 3.0,
        'Bathrooms': 2.0,
        'Square Footage': 1500.0,
        'Acreage': 0.1,
        'Garage': 'Yes',
        'Parking': 'Yes',
        'Basement': 'Finished',
        'Fireplace': 'Yes',
        'Heating': 'forced air',
        'Pool': 'No',
        'Waterfront': 'No'
    }
    
    yearly_predictions1, breakdown1 = predictor.predict_future_price(500000, example1)
    print("\nExample 1: 3BR House in Calgary, AB")
    predictor.print_prediction(breakdown1)
    
    # Example 2: Condo in Vancouver
    example2 = {
        'Province': 'BC',
        'Property Type': 'Condo',
        'Bedrooms': 2.0,
        'Bathrooms': 2.0,
        'Square Footage': 900.0,
        'Acreage': 0.0,
        'Garage': 'No',
        'Parking': 'Yes',
        'Basement': '',
        'Fireplace': 'No',
        'Heating': 'heat pump',
        'Pool': 'No',
        'Waterfront': 'No'
    }
    
    yearly_predictions2, breakdown2 = predictor.predict_future_price(750000, example2)
    print("\nExample 2: 2BR Condo in Vancouver, BC")
    predictor.print_prediction(breakdown2)
    
    # Example 3: Luxury home in West Kelowna
    example3 = {
        'Province': 'BC',
        'Property Type': 'Single Family',
        'Bedrooms': 5.0,
        'Bathrooms': 4.0,
        'Square Footage': 3500.0,
        'Acreage': 0.3,
        'Garage': 'Yes',
        'Parking': 'Yes',
        'Basement': 'Finished',
        'Fireplace': 'Yes',
        'Heating': 'forced air',
        'Pool': 'No',
        'Waterfront': 'No'
    }
    
    yearly_predictions3, breakdown3 = predictor.predict_future_price(1200000, example3)
    print("\nExample 3: 5BR Luxury House in West Kelowna, BC")
    predictor.print_prediction(breakdown3)
    
    print("\n" + "="*80)
    print("STATISTICAL JUSTIFICATION")
    print("="*80)
    print("""
This decomposition is statistically valid because:

1. SEPARABILITY: Market trends (captured by HPI) and house characteristics 
   (captured by features) operate through different mechanisms:
   - HPI reflects macroeconomic factors: interest rates, employment, GDP growth
   - House features reflect microeconomic preferences: quality, location, amenities

2. TIME INVARIANCE: House features (bedrooms, square footage) don't change over time,
   but their market value does. By modeling features separately, we capture their
   relative value independent of market timing.

3. MULTIPLICATIVE MODEL: The formula future_price = current_price × market_growth × 
   house_multiplier assumes percentage changes compound, which aligns with how 
   real estate markets actually behave.

4. CROSS-SECTIONAL VS TIME-SERIES: We avoid the pitfall of treating house listings
   as a time series (they're not - they're a snapshot). Instead, we use them to
   understand relative pricing at a point in time.

5. HEDONIC THEORY: The hedonic pricing model is well-established in economics for
   decomposing goods into constituent characteristics and valuing each separately.

This approach is more robust than naive time-series forecasting per house, which
would require historical data we don't have and would conflate market trends with
house-specific factors.
""")
    
    print("="*80)
    print("SYSTEM READY FOR PREDICTIONS")
    print("="*80)
