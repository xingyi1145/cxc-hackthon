"""
Housing listing scraper using Gemini AI to extract property information.
Works with any real estate website URL - Gemini extracts the relevant data.
"""
import os
import json
from typing import Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class ListingScraper:
    """Scraper for extracting housing listing information from URLs using Gemini AI"""
    
    def __init__(self):
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not found. Scraper will not work.")
        else:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def scrape(self, url: str) -> Dict:
        """
        Extract housing listing information from a URL using Gemini AI.
        
        Args:
            url: The listing URL to analyze (Zillow, Redfin, Realtor.com, etc.)
            
        Returns:
            Dictionary containing extracted property information
        """
        try:
            print(f"[GEMINI SCRAPER] Analyzing URL with Gemini: {url}")
            
            if not hasattr(self, 'client'):
                return {
                    'error': 'Gemini API key not configured',
                    'url': url,
                    'price': None,
                    'location': None,
                }
            
            # Ask Gemini to analyze the URL directly
            property_data = self._extract_with_gemini(url)
            
            # Add the URL to the response
            property_data['url'] = url
            property_data['source'] = 'gemini-scraper'
            
            print(f"[GEMINI SCRAPER] Extracted: {property_data.get('location', 'Unknown')} - {property_data.get('price', 'N/A')}")
            
            return property_data
            
        except Exception as e:
            print(f"[GEMINI SCRAPER] Error: {e}")
            return {
                'error': str(e),
                'url': url,
                'price': None,
                'location': None,
            }
    
    def _extract_with_gemini(self, url: str) -> Dict:
        """
        Use Gemini AI to analyze and extract property information from a URL.
        
        Args:
            url: The property listing URL
            
        Returns:
            Dictionary with extracted property data
        """
        try:
            prompt = f"""Analyze this real estate listing URL and extract all property information.

URL: {url}

Visit the webpage and extract the following information in JSON format. If a field is not found, use null.

Required fields:
- price: The listing price as a formatted string (e.g., "$500,000")
- price_raw: The listing price as an integer (e.g., 500000)
- location: Full location string (e.g., "Seattle, WA 98101")
- address: Street address
- city: City name
- state: State or province code (e.g., "WA", "BC")
- zip_code: Postal/ZIP code

Property details:
- bedrooms: Number of bedrooms (float)
- bathrooms: Number of bathrooms (float)
- square_feet: Square footage (float)
- acreage: Lot size in acres (float)
- property_type: Type (e.g., "Single Family", "Condo", "Townhome")
- year_built: Year the property was built (integer)

Features (use "Yes" or "No" when found, null otherwise):
- garage: Does it have a garage?
- parking: Does it have parking?
- basement: Basement type (e.g., "Finished", "Partial", "No basement", or null)
- fireplace: Does it have a fireplace?
- heating: Heating type (e.g., "forced air", "heat pump")
- pool: Does it have a pool?
- waterfront: Is it waterfront?

Return ONLY valid JSON with no additional text or explanation. Use this exact format:
{{
  "price": "$XXX,XXX",
  "price_raw": 000000,
  "location": "City, State ZIP",
  "address": "123 Street",
  "city": "City",
  "state": "ST",
  "zip_code": "00000",
  "bedrooms": 0.0,
  "bathrooms": 0.0,
  "square_feet": 0.0,
  "acreage": 0.0,
  "property_type": "Type",
  "year_built": 0000,
  "garage": "Yes/No",
  "parking": "Yes/No",
  "basement": "Type",
  "fireplace": "Yes/No",
  "heating": "type",
  "pool": "Yes/No",
  "waterfront": "Yes/No"
}}"""

            # Call Gemini
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            config = types.GenerateContentConfig(
                tools=[grounding_tool]
            )

            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=config
            )
            
            # Parse the response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                # Remove ```json or ``` at start and ``` at end
                lines = response_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_text = '\n'.join(lines)
            
            # Parse JSON
            property_data = json.loads(response_text)
            
            # Ensure all fields exist
            default_data = {
                'price': None,
                'price_raw': None,
                'location': None,
                'address': None,
                'city': None,
                'state': None,
                'zip_code': None,
                'bedrooms': None,
                'bathrooms': None,
                'square_feet': None,
                'acreage': None,
                'property_type': None,
                'year_built': None,
                'garage': None,
                'parking': None,
                'basement': None,
                'fireplace': None,
                'heating': None,
                'pool': None,
                'waterfront': None,
            }
            
            # Merge with defaults
            for key in default_data:
                if key not in property_data or property_data[key] in ['', 'null', 'N/A']:
                    property_data[key] = default_data[key]
            
            return property_data
            
        except json.JSONDecodeError as e:
            print(f"[GEMINI] JSON parse error: {e}")
            print(f"[GEMINI] Response text: {response_text[:500]}")
            return {
                'error': 'Failed to parse property data from AI response',
                'price': None,
                'location': 'Unable to extract location',
            }
        except Exception as e:
            print(f"[GEMINI] Gemini extraction error: {e}")
            return {
                'error': f'AI extraction failed: {str(e)}',
                'price': None,
                'location': 'Unable to extract location',
            }

