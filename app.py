import json
import os
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# --- Configuration ---
app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///housing_stress.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000), debug=True)