"""
Listing model for housing properties
"""
from dataclasses import dataclass
from typing import Optional, Dict
import os


@dataclass
class Listing:
    """Represents a real estate listing"""
    
    url: Optional[str] = None
    price: Optional[str] = None
    price_raw: Optional[int] = None
    location: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Province for Canada
    zip_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Property characteristics
    property_type: Optional[str] = None
    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    square_feet: Optional[float] = None  # Square Footage
    acreage: Optional[float] = None  # Lot size in acres
    year_built: Optional[int] = None
    
    # Features (matching dataset columns)
    garage: Optional[str] = None  # Yes/No
    parking: Optional[str] = None  # Yes/No
    basement: Optional[str] = None  # Finished/Partial/No basement
    exterior: Optional[str] = None
    fireplace: Optional[str] = None  # Yes/No
    heating: Optional[str] = None
    flooring: Optional[str] = None
    roof: Optional[str] = None
    waterfront: Optional[str] = None  # Yes/No
    sewer: Optional[str] = None
    pool: Optional[str] = None  # Yes/No
    garden: Optional[str] = None  # Yes/No
    balcony: Optional[str] = None  # Yes/No
    
    # Metadata
    source: Optional[str] = None
    error: Optional[str] = None
    
    # Predictions
    predicted_prices: Optional[Dict[int, float]] = None  # {year: price} for years 1-5
    prediction_breakdown: Optional[Dict] = None
    
    def to_dict(self):
        """Convert listing to dictionary"""
        return {
            'url': self.url,
            'price': self.price,
            'price_raw': self.price_raw,
            'location': self.location,
            'address': self.address,
            'city': self.city,
            'state': self.state,
            'zip_code': self.zip_code,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'property_type': self.property_type,
            'bedrooms': self.bedrooms,
            'bathrooms': self.bathrooms,
            'square_feet': self.square_feet,
            'acreage': self.acreage,
            'year_built': self.year_built,
            'garage': self.garage,
            'parking': self.parking,
            'basement': self.basement,
            'exterior': self.exterior,
            'fireplace': self.fireplace,
            'heating': self.heating,
            'flooring': self.flooring,
            'roof': self.roof,
            'waterfront': self.waterfront,
            'sewer': self.sewer,
            'pool': self.pool,
            'garden': self.garden,
            'balcony': self.balcony,
            'source': self.source,
            'error': self.error,
            'predicted_prices': self.predicted_prices,
            'prediction_breakdown': self.prediction_breakdown
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create listing from dictionary"""
        return cls(
            url=data.get('url', ''),
            price=data.get('price'),
            price_raw=data.get('price_raw'),
            location=data.get('location'),
            address=data.get('address'),
            city=data.get('city'),
            state=data.get('state'),
            zip_code=data.get('zip_code'),
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            property_type=data.get('property_type'),
            bedrooms=data.get('bedrooms'),
            bathrooms=data.get('bathrooms'),
            square_feet=data.get('square_feet'),
            acreage=data.get('acreage') or data.get('lot_size'),  # Handle both names
            year_built=data.get('year_built'),
            garage=data.get('garage'),
            parking=data.get('parking'),
            basement=data.get('basement'),
            exterior=data.get('exterior'),
            fireplace=data.get('fireplace'),
            heating=data.get('heating'),
            flooring=data.get('flooring'),
            roof=data.get('roof'),
            waterfront=data.get('waterfront'),
            sewer=data.get('sewer'),
            pool=data.get('pool'),
            garden=data.get('garden'),
            balcony=data.get('balcony'),
            source=data.get('source'),
            error=data.get('error'),
            predicted_prices=data.get('predicted_prices'),
            prediction_breakdown=data.get('prediction_breakdown')
        )
    
    def is_valid(self) -> bool:
        """Check if listing has essential information"""
        return self.price_raw is not None and self.location is not None and self.error is None
    
    def predict_future_prices(self, predictor=None, model_path='models/predictor.pkl'):
        """
        Predict future prices for this listing using the price predictor
        
        Args:
            predictor: Optional HousePricePredictor instance (will create if None)
            model_path: Path to the trained model
            
        Returns:
            bool: True if prediction was successful, False otherwise
        """
        if not self.price_raw or not self.is_valid():
            self.error = "Cannot predict: missing essential data"
            return False
        
        try:
            # Import here to avoid circular dependencies
            from house_price_predictor import HousePricePredictor
            
            # Initialize predictor if not provided
            if predictor is None:
                predictor = HousePricePredictor()
                
                # Load the trained model
                if os.path.exists(model_path):
                    predictor.load_models(model_path)
                else:
                    self.error = f"Model not found at {model_path}"
                    return False
            
            # Map state to province
            province_mapping = {
                'BC': 'BC', 'AB': 'AB', 'SK': 'SK', 'MB': 'MB',
                'ON': 'ON', 'QC': 'QC', 'NB': 'NB', 'NS': 'NS',
                'PE': 'PE', 'NL': 'NL', 'YT': 'YT', 'NT': 'NT', 'NU': 'NU'
            }
            
            # Build house features dict - only include fields that have actual data
            house_features = {}
            
            # Province (state)
            if self.state and self.state in province_mapping:
                house_features['Province'] = province_mapping[self.state]
            
            # Property characteristics
            if self.property_type:
                house_features['Property Type'] = self.property_type
            if self.bedrooms is not None:
                house_features['Bedrooms'] = float(self.bedrooms)
            if self.bathrooms is not None:
                house_features['Bathrooms'] = float(self.bathrooms)
            if self.square_feet is not None:
                house_features['Square Footage'] = float(self.square_feet)
            if self.acreage is not None:
                house_features['Acreage'] = float(self.acreage)
            
            # Features - only include if present
            if self.garage:
                house_features['Garage'] = self.garage
            if self.parking:
                house_features['Parking'] = self.parking
            if self.basement:
                house_features['Basement'] = self.basement
            if self.fireplace:
                house_features['Fireplace'] = self.fireplace
            if self.heating:
                house_features['Heating'] = self.heating
            if self.pool:
                house_features['Pool'] = self.pool
            if self.waterfront:
                house_features['Waterfront'] = self.waterfront
            
            # Make prediction
            yearly_predictions, breakdown = predictor.predict_future_price(
                self.price_raw, 
                house_features
            )
            
            # Store predictions
            self.predicted_prices = yearly_predictions
            self.prediction_breakdown = breakdown
            
            return True
            
        except Exception as e:
            self.error = f"Prediction error: {str(e)}"
            return False
    
    def get_5_year_prediction(self) -> Optional[float]:
        """Get the 5-year predicted price"""
        if self.predicted_prices and 5 in self.predicted_prices:
            return self.predicted_prices[5]
        return None
    
    def get_total_appreciation(self) -> Optional[float]:
        """Get the total appreciation percentage over 5 years"""
        if self.prediction_breakdown:
            return self.prediction_breakdown.get('total_appreciation')
        return None
    
    def __str__(self):
        """String representation of listing"""
        if self.error:
            return f"Listing({self.url}) - Error: {self.error}"
        pred_str = f" | 5yr: ${self.get_5_year_prediction():,.0f}" if self.predicted_prices else ""
        return f"Listing({self.location} - {self.price}{pred_str})"
