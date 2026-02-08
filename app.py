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
from google import genai
from google.genai import types
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
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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

class ChatMessage(db.Model):
    """Database model for chat history with AI advisor"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)  # User's message
    response = db.Column(db.Text, nullable=False)  # AI's response
    parameters = db.Column(db.Text)  # JSON string of financial/priority parameters
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def to_dict(self):
        """Convert database model to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'message': self.message,
            'response': self.response,
            'parameters': json.loads(self.parameters) if self.parameters else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<ChatMessage {self.id} - User {self.user_id}>'

class ViewedListing(db.Model):
    """Database model for tracking listings that users view"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    listing_id = db.Column(db.Integer, db.ForeignKey('saved_listing.id'), nullable=False)
    viewed_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def to_dict(self):
        """Convert database model to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'listing_id': self.listing_id,
            'viewed_at': self.viewed_at.isoformat() if self.viewed_at else None
        }
    
    def __repr__(self):
        return f'<ViewedListing {self.id} - User {self.user_id} viewed Listing {self.listing_id}>'

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
    redirect_uri = "http://127.0.0.1:5000/callback"
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
            
        return redirect("http://127.0.0.1:3000/")
        
    except Exception as e:
        print(f"Login error: {e}")
        flash("Authentication failed.")
        return redirect("http://127.0.0.1:3000/")

@app.route("/api/user")
def get_user():
    """Return the logged-in user session info as JSON"""
    user = session.get("user")
    if user:
        return jsonify(user)
    return jsonify(None), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": "http://127.0.0.1:5000/",
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
        
        print(f"[SCRAPE] Received request for URL: {url}")
        
        if not url:
            print("[SCRAPE] Error: No URL provided")
            return jsonify({"error": "URL is required"}), 400
        
        # Scrape the listing
        print(f"[SCRAPE] Starting scrape for: {url}")
        scraped_data = listing_scraper.scrape(url)
        print(f"[SCRAPE] Successfully scraped: {scraped_data.get('location', 'Unknown location')}")
        
        return jsonify(scraped_data)
        
    except Exception as e:
        print(f"[SCRAPE] Error: {e}")
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
        
        print(f"[SAVE] Received request to save listing: {listing_data.get('location', 'Unknown')}")
        
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
        
        print(f"[SAVE] Successfully saved listing ID: {saved_listing.id}")
        
        return jsonify({
            "success": True,
            "listing": saved_listing.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[SAVE] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat-history", methods=["GET"])
def get_chat_history():
    """API endpoint to retrieve chat history for a user"""
    try:
        user_id = request.args.get('user_id', 1, type=int)  # Default to user 1
        limit = request.args.get('limit', 50, type=int)  # Default to 50 messages
        
        messages = db.session.execute(
            db.select(ChatMessage)
            .filter_by(user_id=user_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        ).scalars().all()
        
        return jsonify({
            "success": True,
            "messages": [msg.to_dict() for msg in messages]
        })
        
    except Exception as e:
        print(f"[CHAT HISTORY] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/viewed-listings", methods=["GET", "POST"])
def viewed_listings():
    """API endpoint to track and retrieve viewed listings"""
    if request.method == "POST":
        # Track a viewed listing
        try:
            data = request.get_json()
            user_id = data.get('user_id', 1)  # Default to user 1
            listing_id = data.get('listing_id')
            
            if not listing_id:
                return jsonify({"error": "listing_id is required"}), 400
            
            # Check if already viewed recently (within last hour)
            recent_view = db.session.execute(
                db.select(ViewedListing)
                .filter_by(user_id=user_id, listing_id=listing_id)
                .filter(ViewedListing.viewed_at > db.func.datetime('now', '-1 hour'))
            ).scalar_one_or_none()
            
            if not recent_view:
                viewed = ViewedListing(
                    user_id=user_id,
                    listing_id=listing_id
                )
                db.session.add(viewed)
                db.session.commit()
                print(f"[VIEWED] User {user_id} viewed listing {listing_id}")
            
            return jsonify({"success": True})
            
        except Exception as e:
            db.session.rollback()
            print(f"[VIEWED] Error: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # GET: Retrieve viewed listings
        try:
            user_id = request.args.get('user_id', 1, type=int)
            limit = request.args.get('limit', 50, type=int)
            
            viewed = db.session.execute(
                db.select(ViewedListing)
                .filter_by(user_id=user_id)
                .order_by(ViewedListing.viewed_at.desc())
                .limit(limit)
            ).scalars().all()
            
            # Get the actual listings
            listing_ids = [v.listing_id for v in viewed]
            listings = db.session.execute(
                db.select(SavedListing)
                .filter(SavedListing.id.in_(listing_ids))
            ).scalars().all()
            
            listings_dict = {l.id: l.to_dict() for l in listings}
            
            result = []
            for v in viewed:
                if v.listing_id in listings_dict:
                    result.append({
                        'viewed_at': v.viewed_at.isoformat(),
                        'listing': listings_dict[v.listing_id]
                    })
            
            return jsonify({
                "success": True,
                "viewed_listings": result
            })
            
        except Exception as e:
            print(f"[VIEWED] Error: {e}")
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

Please provide a comprehensive analysis that includes:
1. **Financial Fit**: Calculate monthly payment estimates, down payment requirements, and how it fits within their budget
2. **Key Considerations**: Analyze how each property aligns with their stated priorities (weighted by importance)
3. **Potential Concerns**: Identify any red flags, risks, or challenges (e.g., short on down payment, market conditions, etc.)
4. **Overall Score**: Provide a score out of 100 based on how well the property matches their profile and priorities

Format your response using markdown with:
- **Bold** for section headers
- • Bullet points for lists
- ⚠️ for warnings/concerns
- Clear, actionable advice

Be specific, use numbers when possible, and provide practical recommendations."""

        # Call Gemini API
        if not gemini_client:
            return jsonify({'error': 'Gemini API not configured'}), 500
        
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        
        response_text = response.text
        
        # Save chat message to database
        try:
            user_id = 1  # Default user for testing, should come from session/auth
            chat_message = ChatMessage(
                user_id=user_id,
                message=user_message,
                response=response_text.strip(),
                parameters=json.dumps(parameters) if parameters else None
            )
            db.session.add(chat_message)
            db.session.commit()
            print(f"[CHAT] Saved message ID: {chat_message.id}")
        except Exception as e:
            print(f"[CHAT] Error saving message: {e}")
            db.session.rollback()
        
        return jsonify({
            "content": response_text.strip(),
            "formatted": True
        })
        
    except Exception as e:
        print(f"Chat API error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
