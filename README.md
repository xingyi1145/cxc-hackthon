# A-list housings

A comprehensive platform combining a user feedback-driven web application for housing affordability with a statistically robust machine learning system for 5-year housing price forecasts.

## 🌟 Solution Overview

This project consists of two integrated components:

1.  **Affordability Dashboard (Web App)**:
    *   **Secure Authentication**: User login via Auth0.
    *   **Real-time Scraping**: Extract listing details from URLs.
    *   **AI Financial Advisor**: Gemini-powered chat interface providing personalized financial guidance.
2.  **Price Prediction Engine (ML System)**:
    *   **Market Growth Model**: SARIMA forecasts of the New Housing Price Index (HPI).
    *   **Hedonic Pricing**: XGBoost model incorporating 14+ house features (Province, Sqft, etc.).
    *   **5-Year Forecast**: Detailed year-by-year price trajectory analysis.

---

## 🚀 Quick Start

### 1. Prerequisites
*   Python 3.11+
*   PostgreSQL or SQLite (configured for SQLite by default)
*   Auth0 Account
*   Gemini API Key

### 2. Installation

Clone the repo and set up the environment:

```bash
git clone <repository-url>
cd cxc-hackthon
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# OR using uv
uv sync
```

### 3. Configuration

Create a `.env` file from the template:
```env
APP_SECRET_KEY=your_secret_key
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_DOMAIN=your_domain.auth0.com
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Running the Application

**Web Dashboard:**
```bash
python app.py
```
Access at `http://localhost:3000`.

**Prediction CLI:**
```bash
# Predict price for a specific property
python predict.py \
  --price 750000 \
  --province BC \
  --bedrooms 3 \
  --bathrooms 2 \
  --sqft 2000
```

---

## 📊 Features & Architecture

### Web Application (`app.py`)
*   **Tech Stack**: Flask, SQLAlchemy, Auth0, Google Gemini AI.
*   **Key Endpoints**:
    *   `/login` / `/callback`: Auth0 flow.
    *   `/dashboard`: User profile management.
    *   `/api/scrape-listing`: Extracts data from real estate listings.
    *   `/api/chat`: Generates financial advice based on user profile and listing data.

### Prediction System (`house_price_predictor.py`)
A sophisticated forecasting engine utilizing a decomposition approach:
*   **Market Growth**: Modeled via SARIMA on StatCan HPI data (1981-2025).
*   **House Features**: Modeled via XGBoost on 44,000+ listings.
*   **Gradual Correction**: Adjustment algorithm that corrects mispriced assets over a 3-year period.

See [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) for deep dives into the statistical models.

---

## 📂 Project Structure

```
cxc-hackthon/
├── app.py                    # Main Flask Web Application
├── house_price_predictor.py  # Core ML Prediction Logic
├── predict.py                # CLI for making predictions
├── data/                     # Datasets (HPI, Listings)
├── models/                   # Saved ML models (predictor.pkl)
├── templates/                # HTML templates for Web App
├── static/                   # CSS/JS assets
├── requirements.txt          # Python dependencies
├── QUICKSTART.md             # Fast track guide
├── README_PREDICTION.md      # Detailed ML documentation
└── TECHNICAL_SUMMARY.md      # In-depth technical architecture
```

## 📚 Documentation

*   **[QUICKSTART.md](QUICKSTART.md)**: Get up and running in 3 steps.
*   **[README_PREDICTION.md](README_PREDICTION.md)**: Detailed forecast system usage.
*   **[TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md)**: Methodology and model performance metrics.
*   **[house_prediction_demo.ipynb](house_prediction_demo.ipynb)**: Interactive Jupyter notebook.

