import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Car Price Prediction (Advanced ML)",
    page_icon="🚗",
    layout="centered"
)

# Function to load models
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    
    with open(os.path.join(base_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_dir, 'fuel_type_encoder.pkl'), 'rb') as f:
        fuel_enc = pickle.load(f)
    with open(os.path.join(base_dir, 'seller_type_encoder.pkl'), 'rb') as f:
        seller_enc = pickle.load(f)
    with open(os.path.join(base_dir, 'transmission_encoder.pkl'), 'rb') as f:
        trans_enc = pickle.load(f)
    with open(os.path.join(base_dir, 'car_name_encoder.pkl'), 'rb') as f:
        car_name_enc = pickle.load(f)
        
    return model, scaler, fuel_enc, seller_enc, trans_enc, car_name_enc

try:
    model, scaler, fuel_type_encoder, seller_type_encoder, transmission_encoder, car_name_encoder = load_artifacts()
    st.success("✓ Advanced model and preprocessors loaded successfully!")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Load car names
@st.cache_data
def load_car_names():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, '../car_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return sorted(df['Car_Name'].unique().tolist())
    return []

car_names = load_car_names()

st.title("🚗 Car Price Prediction")
st.markdown("---")

# Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        car_name = st.selectbox("Car Name", car_names if car_names else ["Unknown"])
        year = st.number_input("Year", min_value=1990, max_value=2024, value=2015)
        present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.0, step=0.1)
        kms_driven = st.number_input("Kms Driven", min_value=0, value=25000)
    
    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.number_input("Previous Owners", min_value=0, max_value=3, value=0)
        
    submitted = st.form_submit_button("Predict Selling Price", type="primary")

if submitted:
    try:
        # Feature Engineering (same as training)
        current_year = 2024
        car_age = current_year - year
        price_per_year = present_price / (car_age + 1)
        kms_per_year = kms_driven / (car_age + 1)
        
        # Encode categorical variables
        try:
            car_name_encoded = car_name_encoder.transform([car_name])[0]
        except:
            # If car name not in training data, use median encoding
            car_name_encoded = len(car_name_encoder.classes_) // 2
            
        fuel_type_encoded = fuel_type_encoder.transform([fuel_type])[0]
        seller_type_encoded = seller_type_encoder.transform([seller_type])[0]
        transmission_encoded = transmission_encoder.transform([transmission])[0]
        
        # Create feature array
        features = np.array([[
            year, present_price, kms_driven, owner,
            car_name_encoded, fuel_type_encoded, 
            seller_type_encoded, transmission_encoded,
            car_age, price_per_year, kms_per_year
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # ============================================
        # CRITICAL VALIDATION: Selling price logic
        # ============================================
        
        # Rule 1: Selling price must be positive
        prediction = max(prediction, 0.1) # Changed from 100 to 0.1 Lakhs as likely unit is Lakhs
        
        # Rule 2: Selling price CANNOT exceed present price
        min_depreciation_rate = 0.05
        
        if car_age >= 5:
            min_depreciation_rate += 0.05
        if car_age >= 10:
            min_depreciation_rate += 0.10
        if kms_driven > 100000:
            min_depreciation_rate += 0.08
        if owner >= 2:
            min_depreciation_rate += 0.05
            
        max_selling_price = present_price * (1 - min_depreciation_rate)
        
        if prediction > max_selling_price:
            st.warning(f"⚠️ Prediction capped logic applied (Depreciation rules)")
            prediction = max_selling_price
        
        if prediction > present_price * 0.95:
            prediction = present_price * 0.85

        st.markdown("### Result")
        st.success(f"**Predicted Selling Price:** ₹ {prediction:,.2f} Lakhs")
        
        # Debug info expander
        with st.expander("See debug details"):
            st.write(f"Car: {car_name} ({year})")
            st.write(f"Present Price: {present_price}")
            st.write(f"Depreciation: {((present_price - prediction) / present_price * 100):.1f}%")

    except Exception as e:
        st.error(f"Error calculating prediction: {str(e)}")
