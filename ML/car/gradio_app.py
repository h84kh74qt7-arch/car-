import gradio as gr
import pickle
import numpy as np
import pandas as pd
import os

# 1. Load Artifacts
base_dir = os.path.dirname(__file__)

def load_artifacts():
    try:
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
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None, None, None, None

model, scaler, fuel_type_encoder, seller_type_encoder, transmission_encoder, car_name_encoder = load_artifacts()

# Load car names
def load_car_names():
    csv_path = os.path.join(base_dir, '../car_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return sorted(df['Car_Name'].unique().tolist())
    return []

car_names = load_car_names()

# 2. Prediction Function
def predict_price(car_name, year, present_price, kms_driven, fuel_type, seller_type, transmission, owner):
    try:
        if model is None:
            return "Error: Model not loaded."

        # Feature Engineering
        current_year = 2024
        car_age = current_year - year
        price_per_year = present_price / (car_age + 1)
        kms_per_year = kms_driven / (car_age + 1)
        
        # Encoding
        try:
            car_name_encoded = car_name_encoder.transform([car_name])[0]
        except:
            # Handle unseen labels
            car_name_encoded = len(car_name_encoder.classes_) // 2
            
        fuel_type_encoded = fuel_type_encoder.transform([fuel_type])[0]
        seller_type_encoded = seller_type_encoder.transform([seller_type])[0]
        transmission_encoded = transmission_encoder.transform([transmission])[0]
        
        # Create encoded features array
        features = np.array([[
            year, present_price, kms_driven, owner,
            car_name_encoded, fuel_type_encoded, 
            seller_type_encoded, transmission_encoded,
            car_age, price_per_year, kms_per_year
        ]])
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Validation Logic
        prediction = max(prediction, 0.1)
        
        min_depreciation_rate = 0.05
        if car_age >= 5: min_depreciation_rate += 0.05
        if car_age >= 10: min_depreciation_rate += 0.10
        if kms_driven > 100000: min_depreciation_rate += 0.08
        if owner >= 2: min_depreciation_rate += 0.05
            
        max_selling_price = present_price * (1 - min_depreciation_rate)
        
        response_msg = ""
        if prediction > max_selling_price:
            prediction = max_selling_price
            response_msg += " (Capped using depreciation constraints)"
            
        if prediction > present_price * 0.95:
            prediction = present_price * 0.85
            
        return f"₹ {prediction:,.2f} Lakhs{response_msg}"
        
    except Exception as e:
        return f"Error: {str(e)}"

# 3. Gradio Interface
if __name__ == "__main__":
    if not car_names:
        car_names = ["Unknown"]

    inputs = [
        gr.Dropdown(label="Car Name", choices=car_names, value=car_names[0] if car_names else None),
        gr.Number(label="Year", value=2015, precision=0),
        gr.Number(label="Present Price (Lakhs)", value=5.0),
        gr.Number(label="Kms Driven", value=25000, precision=0),
        gr.Dropdown(label="Fuel Type", choices=["Petrol", "Diesel", "CNG"], value="Petrol"),
        gr.Dropdown(label="Seller Type", choices=["Dealer", "Individual"], value="Dealer"),
        gr.Dropdown(label="Transmission", choices=["Manual", "Automatic"], value="Manual"),
        gr.Number(label="Previous Owners", value=0, precision=0)
    ]
    
    output = gr.Textbox(label="Predicted Selling Price")
    
    demo = gr.Interface(
        fn=predict_price,
        inputs=inputs,
        outputs=output,
        title="🚗 Car Price Prediction (Advanced ML)",
        description="Enter the car details to get an estimated selling price based on the advanced ML model.",
        theme="soft"
    )
    
    demo.launch()
