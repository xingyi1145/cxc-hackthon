import json
import os
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for, request, flash, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import google.generativeai as genai
from listing_scraper import ListingScraper
from listing import Listing as ListingModel

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# --- Configuration ---
app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///housing_stress.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)  # Enable CORS for frontend API calls

# Configure Gemini API
GEMINI_API_KEY = env.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize listing scraper
listing_scraper = ListingScraper()

# --- Database Setup ---
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    auth0_id = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=True)
    name = db.Column(db.String(120), nullable=True)
    
    # Financial Profile - To be implemented by other team members
    # target_down_payment = db.Column(db.Float, default=0.0)
    # max_monthly_payment = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f'<User {self.email}>'

class SavedListing(db.Model):
    """Database model for saved property listings"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    price = db.Column(db.String(50))
    price_raw = db.Column(db.Integer)
    location = db.Column(db.String(200))
    address = db.Column(db.String(200))
    city = db.Column(db.String(100))
    state = db.Column(db.String(50))
    zip_code = db.Column(db.String(20))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    property_type = db.Column(db.String(50))
    bedrooms = db.Column(db.Float)
    bathrooms = db.Column(db.Float)
    square_feet = db.Column(db.Float)
    acreage = db.Column(db.Float)
    year_built = db.Column(db.Integer)
    garage = db.Column(db.String(20))
    parking = db.Column(db.String(20))
    basement = db.Column(db.String(50))
    exterior = db.Column(db.String(50))
    fireplace = db.Column(db.String(20))
    heating = db.Column(db.String(50))
    flooring = db.Column(db.String(50))
    roof = db.Column(db.String(50))
    waterfront = db.Column(db.String(20))
    sewer = db.Column(db.String(50))
    pool = db.Column(db.String(20))
    garden = db.Column(db.String(20))
    balcony = db.Column(db.String(20))
    source = db.Column(db.String(50))
    predicted_prices = db.Column(db.Text)  # JSON string
    prediction_breakdown = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def to_dict(self):
        """Convert database model to dictionary"""
        return {
            'id': self.id,
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
            'predicted_prices': json.loads(self.predicted_prices) if self.predicted_prices else None,
            'prediction_breakdown': json.loads(self.prediction_breakdown) if self.prediction_breakdown else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<SavedListing {self.location} - {self.price}>'

# Create tables if they don't exist
with app.app_context():
    db.create_all()

# --- Auth0 Setup ---
oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)

# --- Routes ---

@app.route("/")
def home():
    user_info = session.get('user')
    return render_template(
        "index.html",
        session=session.get("user"),
        pretty=json.dumps(session.get("user"), indent=4)
    )

@app.route("/login")
def login():
    redirect_uri = url_for("callback", _external=True)
    print(f"DEBUG: Calculated Callback URL: {redirect_uri}")
    return oauth.auth0.authorize_redirect(
        redirect_uri=redirect_uri
    )

@app.route("/callback")
def callback():
    try:
        token = oauth.auth0.authorize_access_token()
        user_info = token.get("userinfo")
        
        # Store basic auth info in session
        session["user"] = user_info
        
        # --- DATABASE SYNC LOGIC ---
        auth0_id = user_info.get("sub")
        
        # Check if user exists
        existing_user = db.session.execute(db.select(User).filter_by(auth0_id=auth0_id)).scalar_one_or_none()
        
        if not existing_user:
            # Create new user
            new_user = User(
                auth0_id=auth0_id,
                email=user_info.get("email"),
                name=user_info.get("name")
            )
            db.session.add(new_user)
            db.session.commit()
            print(f"Created new user: {new_user.email}")
        else:
            print(f"User logged in: {existing_user.email}")
            
        return redirect(url_for("dashboard"))
        
    except Exception as e:
        print(f"Login error: {e}")
        flash("Authentication failed.")
        return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("home", _external=True),
                "client_id": env.get("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )

@app.route("/dashboard", methods=["GET"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    
    auth0_id = session["user"].get("sub")
    current_user = db.session.execute(db.select(User).filter_by(auth0_id=auth0_id)).scalar_one()

    # Financial logic removed - managed by other team members

    return render_template("dashboard.html", user=current_user, auth0_info=session["user"])

@app.route("/api/scrape-listing", methods=["POST"])
def scrape_listing():
    """API endpoint for scraping housing listing information from URLs"""
    try:
        data = request.get_json()
        url = data.get("url", "")
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Scrape the listing
        scraped_data = listing_scraper.scrape(url)
        
        return jsonify(scraped_data)
        
    except Exception as e:
        print(f"Scraping API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/listings", methods=["GET"])
def get_listings():
    """Get all saved listings for the current user"""
    try:
        # Get user from session (for now, we'll use a default user ID if not logged in)
        # TODO: Integrate with actual auth system
        user_id = 1  # Default for testing
        
        listings = SavedListing.query.filter_by(user_id=user_id).order_by(SavedListing.created_at.desc()).all()
        return jsonify([listing.to_dict() for listing in listings])
        
    except Exception as e:
        print(f"Get listings error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/listings", methods=["POST"])
def save_listing():
    """Save a listing to the database"""
    try:
        data = request.get_json()
        listing_data = data.get("listing", {})
        
        # Get user from session (for now, we'll use a default user ID if not logged in)
        user_id = 1  # Default for testing
        
        # Create new saved listing
        saved_listing = SavedListing(
            user_id=user_id,
            url=listing_data.get('url'),
            price=listing_data.get('price'),
            price_raw=listing_data.get('price_raw'),
            location=listing_data.get('location'),
            address=listing_data.get('address'),
            city=listing_data.get('city'),
            state=listing_data.get('state'),
            zip_code=listing_data.get('zip_code'),
            latitude=listing_data.get('latitude'),
            longitude=listing_data.get('longitude'),
            property_type=listing_data.get('property_type'),
            bedrooms=listing_data.get('bedrooms'),
            bathrooms=listing_data.get('bathrooms'),
            square_feet=listing_data.get('square_feet'),
            acreage=listing_data.get('acreage'),
            year_built=listing_data.get('year_built'),
            garage=listing_data.get('garage'),
            parking=listing_data.get('parking'),
            basement=listing_data.get('basement'),
            exterior=listing_data.get('exterior'),
            fireplace=listing_data.get('fireplace'),
            heating=listing_data.get('heating'),
            flooring=listing_data.get('flooring'),
            roof=listing_data.get('roof'),
            waterfront=listing_data.get('waterfront'),
            sewer=listing_data.get('sewer'),
            pool=listing_data.get('pool'),
            garden=listing_data.get('garden'),
            balcony=listing_data.get('balcony'),
            source=listing_data.get('source'),
            predicted_prices=json.dumps(listing_data.get('predicted_prices')) if listing_data.get('predicted_prices') else None,
            prediction_breakdown=json.dumps(listing_data.get('prediction_breakdown')) if listing_data.get('prediction_breakdown') else None
        )
        
        db.session.add(saved_listing)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "listing": saved_listing.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Save listing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """API endpoint for Gemini-powered chat with financial analysis"""
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        parameters = data.get("parameters", {})
        
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini API key not configured"}), 500
        
        # Extract parameters
        financial = parameters.get("financial", {})
        priorities = parameters.get("priorities", [])
        listings = parameters.get("listings", [])
        
        # Build context prompt
        prompt = f"""You are an expert AI real estate advisor helping users analyze properties based on their financial profile and priorities.

USER'S FINANCIAL PROFILE:
- Annual Income: ${financial.get('income', 'N/A')}
- Savings/Down Payment: ${financial.get('savings', 'N/A')}
- Monthly Housing Budget: ${financial.get('budget', 'N/A')}
- Credit Score: {financial.get('creditScore', 'N/A')}
- Risk Tolerance: {financial.get('riskTolerance', 50)}/100 ({'Conservative' if financial.get('riskTolerance', 50) < 35 else 'Aggressive' if financial.get('riskTolerance', 50) > 65 else 'Balanced'})

USER'S PRIORITIES (weighted importance):
"""
        
        for priority in priorities:
            prompt += f"- {priority.get('label', '')}: {priority.get('value', 0)}% - {priority.get('description', '')}\n"
        
        if listings:
            prompt += f"\nPROPERTIES TO ANALYZE:\n"
            for listing in listings:
                # Include scraped data in JSON format
                listing_data = {
                    'url': listing.get('url', 'N/A'),
                    'price': listing.get('price', 'N/A'),
                    'price_raw': listing.get('price_raw'),
                    'location': listing.get('location', 'Unknown'),
                    'address': listing.get('address'),
                    'city': listing.get('city'),
                    'state': listing.get('state'),
                    'zip_code': listing.get('zip_code'),
                    'bedrooms': listing.get('bedrooms'),
                    'bathrooms': listing.get('bathrooms'),
                    'square_feet': listing.get('square_feet'),
                    'property_type': listing.get('property_type'),
                    'year_built': listing.get('year_built'),
                    'lot_size': listing.get('lot_size'),
                    'source': listing.get('source', 'unknown')
                }
                
                prompt += f"\nProperty {listing.get('location', 'Unknown')}:\n"
                prompt += f"JSON Data: {json.dumps(listing_data, indent=2)}\n"
                prompt += f"URL: {listing.get('url', 'N/A')}\n"
        
        prompt += f"""
USER'S QUESTION: {user_message}

If the user only provides a greeting, suggest possible questions that they may want to inquire about.

When the user asks about a specific property or mentions a property URL, create a listing for tracking. After your analysis, include a structured listing marker:
SAVE_LISTING_START{{"url": "property_url", "price": "$XXX,XXX", "price_raw": 000000, "location": "City, State", "city": "City", "state": "State", "bedrooms": X, "bathrooms": X, "property_type": "Single Family", "square_feet": XXXX}}SAVE_LISTING_END

Please provide a comprehensive analysis that includes:
1. **Financial Fit**: Calculate monthly payment estimates, down payment requirements, and how it fits within their budget
2. **Key Considerations**: Analyze how each proper  ty aligns with their stated priorities (weighted by importance)
3. **Potential Concerns**: Identify any red flags, risks, or challenges (e.g., short on down payment, market conditions, etc.)
4. **Overall Score**: Provide a score out of 100 based on how well the property matches their profile and priorities

Format your response using markdown with:
- **Bold** for section headers
- • Bullet points for lists
- ⚠️ for warnings/concerns
- Clear, actionable advice

Be specific, use numbers when possible, and provide practical recommendations."""

        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        response_text = response.text
        saved_listings = []
        
        # Check if Gemini wants to save a listing
        if 'SAVE_LISTING_START' in response_text and 'SAVE_LISTING_END' in response_text:
            # Extract listing data
            start_idx = response_text.find('SAVE_LISTING_START') + len('SAVE_LISTING_START')
            end_idx = response_text.find('SAVE_LISTING_END')
            listing_json = response_text[start_idx:end_idx].strip()
            
            try:
                listing_data = json.loads(listing_json)
                
                # Get user from session (default to user 1 for testing)
                user_id = 1
                
                # Create and save listing
                saved_listing = SavedListing(
                    user_id=user_id,
                    url=listing_data.get('url'),
                    price=listing_data.get('price'),
                    price_raw=listing_data.get('price_raw'),
                    location=listing_data.get('location'),
                    city=listing_data.get('city'),
                    state=listing_data.get('state'),
                    property_type=listing_data.get('property_type'),
                    bedrooms=listing_data.get('bedrooms'),
                    bathrooms=listing_data.get('bathrooms'),
                    square_feet=listing_data.get('square_feet'),
                    acreage=listing_data.get('acreage'),
                    source='gemini-created'
                )
                
                db.session.add(saved_listing)
                db.session.commit()
                saved_listings.append(saved_listing.to_dict())
                
                # Remove the save marker from response
                response_text = response_text[:response_text.find('SAVE_LISTING_START')] + response_text[end_idx + len('SAVE_LISTING_END'):]
                
            except Exception as e:
                print(f"Error saving listing from Gemini: {e}")
        
        return jsonify({
            "content": response_text.strip(),
            "formatted": True,
            "saved_listings": saved_listings
        })
        
    except Exception as e:
        print(f"Chat API error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 5001), debug=True)