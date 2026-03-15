import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Airbnb Data Science Project",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_and_prep_data():
    listings = pd.read_csv('listi.csv')
    listings = listings.rename(columns={'room _type': 'room_type', 'ttm avg rate': 'ttm_avg_rate'})
    past = pd.read_csv('past.csv')
    fut = pd.read_csv('fut.csv')
    reviews = pd.read_csv('reviews.csv')

    listings['guests'] = listings['guests'].fillna(1)
    listings['bedrooms'] = listings['bedrooms'].fillna(listings['bedrooms'].median())
    listings['beds'] = listings['beds'].fillna(1)
    listings['baths'] = listings['baths'].fillna(1)
    listings['ttm_avg_rate'] = listings['ttm_avg_rate'].fillna(listings['ttm_avg_rate'].mean())
    listings['room_type'] = listings['room_type'].fillna('Unknown')

    room_dummies = pd.get_dummies(listings['room_type'], prefix='room')
    listings_ml = pd.concat([listings, room_dummies], axis=1)
    
    base_features = ['bedrooms', 'beds', 'baths', 'guests']
    room_features = list(room_dummies.columns)
    features = base_features + room_features

    ml_df = listings_ml.dropna(subset=['ttm_avg_rate'])
    ai = RandomForestRegressor(n_estimators=100, random_state=42)
    ai.fit(ml_df[features], ml_df['ttm_avg_rate'])

    past['date'] = pd.to_datetime(past['date'])
    fut['date'] = pd.to_datetime(fut['date'])

    return listings, past, fut, reviews, ai, features

listings, past, fut, reviews, ai, features = load_and_prep_data()

col_spacer_left, col_hero, col_spacer_right = st.columns([1, 2, 1])

with col_hero:
    try:
        st.image("logo.gif", use_container_width=True)
    except:
        st.info("Logo GIF file not found.")
    
    st.title("Airbnb Data Science Project")
    st.subheader("Exploring Paris Listings & Price Predictions")
    st.markdown("**Data Source:** [AirROI Paris Market Data](https://www.airroi.com/data-portal/markets/paris-france)")
    st.write("Scroll down to explore market trends and use our AI price predictor.")
    
    st.markdown("---")

st.header(" Market Explorer")
st.write("Dive into the Paris Airbnb market with interactive maps and key metrics.")

col_map, col_stats = st.columns([2, 1])

with col_map:
    fig_map = px.scatter_mapbox(
        listings, 
        lat="latitude", lon="longitude", 
        color="ttm_avg_rate", size="guests", 
        color_continuous_scale='Reds',
        hover_name="listing name", 
        zoom=11, height=600,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_stats:
    st.subheader("Market Vitality")
    st.metric("Total Listings", len(listings))
    st.metric("Market Occupancy", f"{listings['ttm occupancy'].mean()*100:.1f}%")
    st.metric("Avg Nightly Price", f"€{listings['ttm_avg_rate'].mean():.2f}")
    
    st.subheader("Inventory")
    fig_pie = px.pie(listings, names='room_type', hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Monthly Revenue Performance")
combined = pd.merge(past, fut, on='date', how='outer', suffixes=('_past', '_fut'))
combined['revenue'] = combined['revenue_past'].fillna(0) + combined['revenue_fut'].fillna(0)
combined['month'] = combined['date'].dt.to_period('M').dt.to_timestamp()
rev_monthly = combined.groupby('month')['revenue'].sum().reset_index()

fig_rev = px.bar(
    rev_monthly, x='month', y='revenue', 
    color='revenue', color_continuous_scale='Blues',
    text_auto='.2s'
)
st.plotly_chart(fig_rev, use_container_width=True)

st.divider()

st.header("✨ AI Recommendations")
col_pref, col_rec = st.columns([1, 2])
with col_pref:
    max_price = st.slider("Max Nightly Price (€)", 0, 2000, 200)
    pref_room_type = st.selectbox("Preferred Room Type", options=listings['room_type'].unique())
    min_guests = st.slider("Minimum Guests", 1, 20, 2)
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 4.0)

with col_rec:
    filtered = listings[
        (listings['ttm_avg_rate'] <= max_price) &
        (listings['room_type'] == pref_room_type) &
        (listings['guests'] >= min_guests) &
        (listings['rating overall'] >= min_rating)
    ].sort_values('rating overall', ascending=False).head(5)
    
    if not filtered.empty:
        for _, row in filtered.iterrows():
            st.info(f"**{row['listing name']}**\n\n€{row['ttm_avg_rate']:.2f}/night — ⭐ {row['rating overall']}")
    else:
        st.warning("No listings match your criteria.")

st.divider()

st.header("🤖 AI Price Predictor")
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Property Features")
    p_rooms = st.number_input("Bedrooms", 0, 10, 1)
    p_beds = st.number_input("Beds", 0, 10, 1)
    p_baths = st.number_input("Bathrooms", 1, 10, 1)
    p_guests = st.number_input("Guests", 1, 20, 2)
    p_type = st.selectbox("Room Type Selection", options=listings['room_type'].unique())

    input_data = pd.DataFrame([[p_rooms, p_beds, p_baths, p_guests]], columns=['bedrooms', 'beds', 'baths', 'guests'])
    for f in features:
        if f.startswith('room_'):
            input_data[f] = 1 if f == f"room_{p_type}" else 0
    input_data = input_data[features]

with col_output:
    st.subheader("Predicted Price")
    pred = ai.predict(input_data)[0]
    st.metric("Estimated Price", f"€{pred:.2f}")
    
    imp_df = pd.DataFrame({'Feature': features, 'Importance': ai.feature_importances_}).sort_values('Importance')
    st.bar_chart(imp_df.set_index('Feature'))

