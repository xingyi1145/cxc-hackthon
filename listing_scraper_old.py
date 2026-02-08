"""
Housing listing scraper for extracting property information from real estate websites.
Supports Zillow, Realtor.com, Redfin, and other major platforms.
"""
import re
import json
import requests
from typing import Dict, Optional
from urllib.parse import urlparse


class ListingScraper:
    """Scraper for extracting housing listing information from URLs"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def scrape(self, url: str) -> Dict:
        """
        Scrape housing listing information from a URL.
        
        Args:
            url: The listing URL to scrape
            
        Returns:
            Dictionary containing scraped property information
        """
        try:
            # Determine which scraper to use based on URL
            domain = urlparse(url).netloc.lower()
            
            if 'zillow.com' in domain:
                return self._scrape_zillow(url)
            elif 'realtor.com' in domain:
                return self._scrape_realtor(url)
            elif 'redfin.com' in domain:
                return self._scrape_redfin(url)
            else:
                # Generic scraper for unknown sites
                return self._scrape_generic(url)
        except Exception as e:
            return {
                'error': str(e),
                'url': url,
                'price': None,
                'location': None,
                'raw_data': {}
            }
    
    def _scrape_zillow(self, url: str) -> Dict:
        """Scrape Zillow listing"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # Extract price
            price_match = re.search(r'"price":\s*(\d+)', html)
            price = price_match.group(1) if price_match else None
            
            # Extract address
            address_match = re.search(r'"streetAddress":\s*"([^"]+)"', html)
            city_match = re.search(r'"addressLocality":\s*"([^"]+)"', html)
            state_match = re.search(r'"addressRegion":\s*"([^"]+)"', html)
            zip_match = re.search(r'"postalCode":\s*"([^"]+)"', html)
            
            address_parts = []
            if address_match:
                address_parts.append(address_match.group(1))
            if city_match:
                address_parts.append(city_match.group(1))
            if state_match:
                address_parts.append(state_match.group(1))
            if zip_match:
                address_parts.append(zip_match.group(1))
            
            location = ', '.join(address_parts) if address_parts else None
            
            # Extract property details
            beds_match = re.search(r'"numberOfBedrooms":\s*"?(\d+)"?', html)
            baths_match = re.search(r'"numberOfBathroomsTotal":\s*"?([\d.]+)"?', html)
            sqft_match = re.search(r'"floorSize":\s*\{[^}]*"value":\s*(\d+)', html)
            
            # Extract property type
            prop_type_match = re.search(r'"@type":\s*"([^"]+)"', html)
            property_type = prop_type_match.group(1) if prop_type_match else None
            
            # Extract year built
            year_built_match = re.search(r'"yearBuilt":\s*"?(\d{4})"?', html)
            
            # Extract lot size
            lot_size_match = re.search(r'"lotSize":\s*\{[^}]*"value":\s*([\d.]+)', html)
            
            return {
                'url': url,
                'price': f'${int(price):,}' if price else None,
                'price_raw': int(price) if price else None,
                'location': location or 'Location not found',
                'address': address_match.group(1) if address_match else None,
                'city': city_match.group(1) if city_match else None,
                'state': state_match.group(1) if state_match else None,
                'zip_code': zip_match.group(1) if zip_match else None,
                'bedrooms': int(beds_match.group(1)) if beds_match else None,
                'bathrooms': float(baths_match.group(1)) if baths_match else None,
                'square_feet': int(sqft_match.group(1)) if sqft_match else None,
                'property_type': property_type,
                'year_built': int(year_built_match.group(1)) if year_built_match else None,
                'lot_size': float(lot_size_match.group(1)) if lot_size_match else None,
                'source': 'zillow',
                'raw_data': {
                    'html_length': len(html),
                    'has_price': price is not None,
                    'has_location': location is not None
                }
            }
        except Exception as e:
            return {
                'error': f'Zillow scraping error: {str(e)}',
                'url': url,
                'price': None,
                'location': None,
                'raw_data': {}
            }
    
    def _scrape_realtor(self, url: str) -> Dict:
        """Scrape Realtor.com listing"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # Extract price
            price_match = re.search(r'"price":\s*(\d+)', html) or re.search(r'\$([\d,]+)', html)
            price = price_match.group(1).replace(',', '') if price_match else None
            
            # Extract address
            address_match = re.search(r'"streetAddress":\s*"([^"]+)"', html)
            city_match = re.search(r'"addressLocality":\s*"([^"]+)"', html)
            state_match = re.search(r'"addressRegion":\s*"([^"]+)"', html)
            
            address_parts = []
            if address_match:
                address_parts.append(address_match.group(1))
            if city_match:
                address_parts.append(city_match.group(1))
            if state_match:
                address_parts.append(state_match.group(1))
            
            location = ', '.join(address_parts) if address_parts else None
            
            # Extract property details
            beds_match = re.search(r'"bedrooms":\s*"?(\d+)"?', html) or re.search(r'(\d+)\s+bed', html, re.I)
            baths_match = re.search(r'"bathrooms":\s*"?([\d.]+)"?', html) or re.search(r'(\d+\.?\d*)\s+bath', html, re.I)
            sqft_match = re.search(r'"squareFootage":\s*"?(\d+)"?', html) or re.search(r'([\d,]+)\s*sq\.?\s*ft', html, re.I)
            
            return {
                'url': url,
                'price': f'${int(price):,}' if price else None,
                'price_raw': int(price) if price else None,
                'location': location or 'Location not found',
                'address': address_match.group(1) if address_match else None,
                'city': city_match.group(1) if city_match else None,
                'state': state_match.group(1) if state_match else None,
                'bedrooms': int(beds_match.group(1)) if beds_match else None,
                'bathrooms': float(baths_match.group(1)) if baths_match else None,
                'square_feet': int(sqft_match.group(1).replace(',', '')) if sqft_match else None,
                'source': 'realtor.com',
                'raw_data': {
                    'html_length': len(html),
                    'has_price': price is not None,
                    'has_location': location is not None
                }
            }
        except Exception as e:
            return {
                'error': f'Realtor.com scraping error: {str(e)}',
                'url': url,
                'price': None,
                'location': None,
                'raw_data': {}
            }
    
    def _scrape_redfin(self, url: str) -> Dict:
        """Scrape Redfin listing"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # Extract price
            price_match = re.search(r'"price":\s*(\d+)', html) or re.search(r'\$([\d,]+)', html)
            price = price_match.group(1).replace(',', '') if price_match else None
            
            # Extract address
            address_match = re.search(r'"streetAddress":\s*"([^"]+)"', html)
            city_match = re.search(r'"addressLocality":\s*"([^"]+)"', html)
            state_match = re.search(r'"addressRegion":\s*"([^"]+)"', html)
            
            address_parts = []
            if address_match:
                address_parts.append(address_match.group(1))
            if city_match:
                address_parts.append(city_match.group(1))
            if state_match:
                address_parts.append(state_match.group(1))
            
            location = ', '.join(address_parts) if address_parts else None
            
            # Extract property details
            beds_match = re.search(r'"bedrooms":\s*"?(\d+)"?', html) or re.search(r'(\d+)\s+bed', html, re.I)
            baths_match = re.search(r'"bathrooms":\s*"?([\d.]+)"?', html) or re.search(r'(\d+\.?\d*)\s+bath', html, re.I)
            sqft_match = re.search(r'"squareFootage":\s*"?(\d+)"?', html) or re.search(r'([\d,]+)\s*sq\.?\s*ft', html, re.I)
            
            return {
                'url': url,
                'price': f'${int(price):,}' if price else None,
                'price_raw': int(price) if price else None,
                'location': location or 'Location not found',
                'address': address_match.group(1) if address_match else None,
                'city': city_match.group(1) if city_match else None,
                'state': state_match.group(1) if state_match else None,
                'bedrooms': int(beds_match.group(1)) if beds_match else None,
                'bathrooms': float(baths_match.group(1)) if baths_match else None,
                'square_feet': int(sqft_match.group(1).replace(',', '')) if sqft_match else None,
                'source': 'redfin',
                'raw_data': {
                    'html_length': len(html),
                    'has_price': price is not None,
                    'has_location': location is not None
                }
            }
        except Exception as e:
            return {
                'error': f'Redfin scraping error: {str(e)}',
                'url': url,
                'price': None,
                'location': None,
                'raw_data': {}
            }
    
    def _scrape_generic(self, url: str) -> Dict:
        """Generic scraper for unknown sites - attempts basic extraction"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # Try to extract price
            price_patterns = [
                r'\$([\d,]+)',
                r'"price":\s*(\d+)',
                r'price["\']?\s*[:=]\s*["\']?\$?([\d,]+)',
            ]
            
            price = None
            for pattern in price_patterns:
                match = re.search(pattern, html)
                if match:
                    price = match.group(1).replace(',', '')
                    break
            
            # Try to extract address/location
            address_patterns = [
                r'"streetAddress":\s*"([^"]+)"',
                r'"addressLocality":\s*"([^"]+)"',
            ]
            
            location_parts = []
            for pattern in address_patterns:
                match = re.search(pattern, html)
                if match:
                    location_parts.append(match.group(1))
            
            location = ', '.join(location_parts) if location_parts else 'Location not found'
            
            return {
                'url': url,
                'price': f'${int(price):,}' if price else None,
                'price_raw': int(price) if price else None,
                'location': location,
                'source': 'generic',
                'raw_data': {
                    'html_length': len(html),
                    'has_price': price is not None,
                    'has_location': len(location_parts) > 0
                }
            }
        except Exception as e:
            return {
                'error': f'Generic scraping error: {str(e)}',
                'url': url,
                'price': None,
                'location': 'Location not found',
                'raw_data': {}
            }

