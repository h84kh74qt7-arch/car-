from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the trained model and preprocessors
print("Loading advanced model and preprocessors...")
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('fuel_type_encoder.pkl', 'rb') as file:
    fuel_type_encoder = pickle.load(file)

with open('seller_type_encoder.pkl', 'rb') as file:
    seller_type_encoder = pickle.load(file)

with open('transmission_encoder.pkl', 'rb') as file:
    transmission_encoder = pickle.load(file)

with open('car_name_encoder.pkl', 'rb') as file:
    car_name_encoder = pickle.load(file)

# Load car names from CSV for the dropdown
csv_path = '../car_data.csv'
df = pd.read_csv(csv_path)
car_names = sorted(df['Car_Name'].unique().tolist())

print("✓ Advanced model and preprocessors loaded successfully!")
print(f"✓ Model type: {type(model).__name__}")
print(f"✓ Available car models: {len(car_names)}")

@app.route('/')
def index():
    """Render the main prediction form"""
    return render_template('index.html', car_names=car_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with advanced feature engineering"""
    try:
        # Get form data
        car_name = request.form['Car_Name']
        year = int(request.form['Year'])
        present_price = float(request.form['Present_Price'])
        kms_driven = float(request.form['Kms_Driven'])
        fuel_type = request.form['Fuel_Type']
        seller_type = request.form['Seller_Type']
        transmission = request.form['Transmission']
        owner = int(request.form['Owner'])
        
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
        
        # Create feature array (must match training order)
        # Order: Year, Present_Price, Kms_Driven, Owner, Car_Name_Encoded,
        #        Fuel_Type_Encoded, Seller_Type_Encoded, Transmission_Encoded,
        #        Car_Age, Price_Per_Year, Kms_Per_Year
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
        prediction = max(prediction, 100)
        
        # Rule 2: Selling price CANNOT exceed present price
        # Apply minimum depreciation based on age, mileage, and owner count
        min_depreciation_rate = 0.05  # At least 5% depreciation
        
        # Add more depreciation for:
        if car_age >= 5:
            min_depreciation_rate += 0.05
        if car_age >= 10:
            min_depreciation_rate += 0.10
        if kms_driven > 100000:
            min_depreciation_rate += 0.08
        if owner >= 2:
            min_depreciation_rate += 0.05
        
        # Maximum allowed selling price
        max_selling_price = present_price * (1 - min_depreciation_rate)
        
        # Cap the prediction
        if prediction > max_selling_price:
            print(f"⚠️  Prediction capped: ${prediction:.2f} → ${max_selling_price:.2f}")
            prediction = max_selling_price
        
        # Additional sanity check: never exceed 95% of present price
        if prediction > present_price * 0.95:
            prediction = present_price * 0.85  # Use 85% as safe value
        
        # Log the prediction for debugging
        print(f"\n🔍 Prediction Details:")
        print(f"   Car: {car_name} ({year})")
        print(f"   Present Price: ${present_price:,.2f}")
        print(f"   Predicted Selling Price: ${prediction:,.2f}")
        print(f"   Depreciation: {((present_price - prediction) / present_price * 100):.1f}%")
        
        # Format prediction
        prediction_formatted = f"{prediction:,.2f}"
        
        return render_template('resultat.html', 
                             prediction=prediction_formatted,
                             car_name=car_name,
                             year=year)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"Error making prediction: {str(e)}<br><br><a href='/'>Go Back</a>"

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚗 CAR PRICE PREDICTION APP (Advanced ML)")
    print("=" * 60)
    print("\n✓ Server starting...")
    print("✓ Open your browser and go to: http://127.0.0.1:5000")
    print("\n" + "=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
