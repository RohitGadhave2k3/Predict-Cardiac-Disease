import streamlit as st
import pandas as pd
import numpy as np
from preprocessing.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from recommendation.recommender import RestaurantRecommender
from utils.visualization import Visualizer
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Restaurant Ratings & Recommendations",
    page_icon="🍽️",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'recommender' not in st.session_state:
    st.session_state.recommender = None

def load_data():
    """Load and preprocess data"""
    try:
        data = pd.read_csv('data/restaurant_data.csv')
        return data
    except:
        st.error("Error loading data. Please make sure the data file exists.")
        return None

def train_model():
    """Train the rating prediction model"""
    data = load_data()
    if data is None:
        return
    
    processor = DataProcessor()
    feature_cols = ['cuisine_type', 'price_range', 'location', 'service_rating']
    categorical_cols = ['cuisine_type', 'location']
    
    X_train, X_test, y_train, y_test = processor.prepare_data(
        data, 'rating', feature_cols, categorical_cols
    )
    
    model = ModelTrainer(model_type='random_forest')
    model.train(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    st.session_state.model = model
    
    return metrics

def train_recommender():
    """Train the recommendation system"""
    data = load_data()
    if data is None:
        return
    
    recommender = RestaurantRecommender()
    ratings_matrix = recommender.create_user_restaurant_matrix(data)
    recommender.train_collaborative_filtering()
    
    # Prepare restaurant features for content-based filtering
    restaurant_features = pd.get_dummies(data[['cuisine_type', 'price_range', 'location']])
    recommender.train_content_based(restaurant_features)
    
    st.session_state.recommender = recommender

def main():
    st.title("🍽️ Restaurant Ratings & Recommendations")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Rating Prediction", "Recommendations"])
    
    if page == "Home":
        st.write("""
        ## Welcome to Restaurant Ratings & Recommendations System
        
        This application helps you:
        1. Predict restaurant ratings based on various features
        2. Get personalized restaurant recommendations
        3. Analyze restaurant data and trends
        
        Choose a section from the sidebar to get started!
        """)
        
        if st.button("Load Data & Train Models"):
            with st.spinner("Training models..."):
                metrics = train_model()
                train_recommender()
                
                if metrics:
                    st.success("Models trained successfully!")
                    st.write("### Model Performance Metrics")
                    st.write(metrics)
    
    elif page == "Rating Prediction":
        st.header("Restaurant Rating Prediction")
        
        if st.session_state.model is None:
            st.warning("Please train the model first from the Home page.")
            return
        
        # Input form for prediction
        cuisine_type = st.selectbox("Cuisine Type", 
                                  ["Italian", "Japanese", "Indian", "American", "Mexican"])
        price_range = st.slider("Price Range (1-5)", 1, 5, 3)
        location = st.selectbox("Location", 
                              ["Downtown", "Suburb", "City Center", "Business District"])
        service_rating = st.slider("Service Rating (1-5)", 1, 5, 4)
        
        if st.button("Predict Rating"):
            # Prepare input data
            input_data = pd.DataFrame({
                'cuisine_type': [cuisine_type],
                'price_range': [price_range],
                'location': [location],
                'service_rating': [service_rating]
            })
            
            # Make prediction
            prediction = st.session_state.model.predict(input_data)
            st.write(f"### Predicted Rating: {prediction[0]:.1f} ⭐")
    
    elif page == "Recommendations":
        st.header("Restaurant Recommendations")
        
        if st.session_state.recommender is None:
            st.warning("Please train the recommender first from the Home page.")
            return
        
        user_id = st.number_input("Enter User ID", min_value=1, value=1)
        n_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
        
        if st.button("Get Recommendations"):
            recommendations = st.session_state.recommender.hybrid_recommendations(
                user_id, n_recommendations=n_recommendations
            )
            
            if recommendations:
                st.write("### Recommended Restaurants")
                for i, rest_id in enumerate(recommendations, 1):
                    st.write(f"{i}. Restaurant ID: {rest_id}")
            else:
                st.write("No recommendations found for this user.")

if __name__ == "__main__":
    main() 