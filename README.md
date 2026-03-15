# AirBnb-Analysis-Project
An interactive web application for exploring and analyzing Airbnb listings in Paris, featuring market insights, AI-powered price predictions, and data-driven recommendations.

## 📊 Overview

This project analyzes Airbnb market data for Paris, providing:
- Interactive market exploration with maps and statistics
- Historical and projected revenue analysis
- AI-powered price prediction using machine learning
- Personalized listing recommendations based on user preferences

## 🚀 Features

### Market Explorer
- Interactive map visualization of Paris listings
- Key market metrics (total listings, occupancy rates, average prices)
- Room type distribution analysis
- Monthly revenue performance charts

### AI Recommendations
- Filter listings by price, room type, guest capacity, and ratings
- Top 5 recommendations based on user preferences

### AI Price Predictor
- Predict nightly rates using Random Forest regression
- Feature importance visualization
- Input property characteristics (bedrooms, beds, bathrooms, guests, room type)

## 📁 Dataset

The project uses the following CSV files:
- `listi.csv` - Current Airbnb listings data
- `past.csv` - Historical revenue data
- `fut.csv` - Projected future revenue data
- `reviews.csv` - Guest reviews and ratings

**Data Source:** [AirROI Paris Market Data](https://www.airroi.com/data-portal/markets/paris-france)

## 🛠️ Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn

## ▶️ Usage

Run the Streamlit application:

```bash
streamlit run self.py
```

The app will open in your default web browser at `http://localhost:8501`

## � Live Demo

Check out the live version of this application: [Airbnb Analysis Project](https://airbnb-analysis-proj.streamlit.app/)

## �🏗️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas & NumPy
- **Visualization:** Plotly Express
- **Machine Learning:** Scikit-Learn (Random Forest)

## 🤖 Machine Learning Model

The price prediction model uses a Random Forest Regressor trained on listing features including:
- Number of bedrooms, beds, bathrooms
- Guest capacity
- Room type (encoded as dummy variables)

## 📈 Data Analysis

The application provides insights into:
- Market occupancy rates
- Average nightly prices
- Revenue trends over time
- Geographic distribution of listings
- Room type preferences

## 📝 License

This project is for educational and demonstration purposes.
