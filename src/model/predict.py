"""
Simple interface for making house price predictions

Usage:
    python predict.py --price 600000 --province AB --bedrooms 3 --bathrooms 2 --sqft 1800
"""

import argparse
import sys
import os

# Import classes so pickle can find them
import house_price_predictor
from house_price_predictor import HousePricePredictor, HPIForecaster, HedonicPricingModel


def main():
    parser = argparse.ArgumentParser(description='Predict house price 5 years in the future')
    
    # Required arguments
    parser.add_argument('--price', type=float, required=True, 
                        help='Current house price in dollars')
    
    # House features
    parser.add_argument('--province', type=str, default='BC',
                        help='Province (e.g., BC, AB, SK, ON)')
    parser.add_argument('--property-type', type=str, default='Single Family',
                        help='Property type (e.g., Single Family, Condo, Townhome)')
    parser.add_argument('--bedrooms', type=float, default=3.0,
                        help='Number of bedrooms')
    parser.add_argument('--bathrooms', type=float, default=2.0,
                        help='Number of bathrooms')
    parser.add_argument('--sqft', type=float, default=1500.0,
                        help='Square footage')
    parser.add_argument('--acreage', type=float, default=0.1,
                        help='Land acreage')
    parser.add_argument('--garage', type=str, default='Yes',
                        help='Garage (Yes/No)')
    parser.add_argument('--parking', type=str, default='Yes',
                        help='Parking (Yes/No)')
    parser.add_argument('--basement', type=str, default='Finished',
                        help='Basement (Finished/Partial/No basement)')
    parser.add_argument('--fireplace', type=str, default='No',
                        help='Fireplace (Yes/No)')
    parser.add_argument('--heating', type=str, default='forced air',
                        help='Heating type')
    parser.add_argument('--pool', type=str, default='No',
                        help='Pool (Yes/No)')
    parser.add_argument('--waterfront', type=str, default='No',
                        help='Waterfront (Yes/No)')
    
    # Model path
    parser.add_argument('--model-path', type=str, default='models/predictor.pkl',
                        help='Path to trained model')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain models before predicting')
    parser.add_argument('--plot', action='store_true',
                        help='Generate price trajectory plot')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load or train models
    if args.retrain or not os.path.exists(args.model_path):
        print("Training models...")
        predictor.train()
        predictor.save_models(args.model_path)
    else:
        print(f"Loading models from {args.model_path}...")
        predictor.load_models(args.model_path)
    
    # Prepare house features
    house_features = {
        'Province': args.province,
        'Property Type': args.property_type,
        'Bedrooms': args.bedrooms,
        'Bathrooms': args.bathrooms,
        'Square Footage': args.sqft,
        'Acreage': args.acreage,
        'Garage': args.garage,
        'Parking': args.parking,
        'Basement': args.basement,
        'Fireplace': args.fireplace,
        'Heating': args.heating,
        'Pool': args.pool,
        'Waterfront': args.waterfront
    }
    
    # Make prediction
    yearly_predictions, breakdown = predictor.predict_future_price(args.price, house_features)
    
    # Print results
    predictor.print_prediction(breakdown)
    
    # Generate plot if requested
    if args.plot:
        predictor.plot_prediction(breakdown, 'price_trajectory.png')
    
    # Summary
    print(f"\n📊 QUICK SUMMARY")
    print(f"Starting Price: ${args.price:,.0f}")
    print(f"Year 1: ${yearly_predictions[1]:,.0f}  |  Year 2: ${yearly_predictions[2]:,.0f}  |  Year 3: ${yearly_predictions[3]:,.0f}")
    print(f"Year 4: ${yearly_predictions[4]:,.0f}  |  Year 5: ${yearly_predictions[5]:,.0f}")
    print(f"5-Year Change: {breakdown['total_appreciation']:+.2f}%")


if __name__ == "__main__":
    main()
