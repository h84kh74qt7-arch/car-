import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# 1. THE FUNCTION TO COPY TO YOUR NOTEBOOK
# ==========================================
def plot_performance_curves(model, X_train, y_train, X_test, y_test):
    """
    Generates and displays performance curves for model evaluation.
    Copy this function into your Jupyter Notebook.
    """
    print("\n📊 Generating Performance Curves...")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Configure figure
    plt.figure(figsize=(20, 15))
    plt.style.use('seaborn-v0_8-whitegrid')

    # A. Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='#2c3e50')
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Performance: Actual vs Predicted', fontsize=14, loc='left')
    
    # B. Residuals Distribution
    residuals = y_test - y_pred
    plt.subplot(2, 2, 2)
    sns.histplot(residuals, kde=True, color='#e74c3c')
    plt.axvline(x=0, color='k', linestyle='--', lw=1)
    plt.xlabel('Error (Actual - Predicted)')
    plt.title('Error Distribution (Residuals)', fontsize=14, loc='left')

    # C. Learning Curve
    plt.subplot(2, 2, 3)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color='#2980b9', label='Train Score')
    plt.plot(train_sizes, val_mean, 'o-', color='#27ae60', label='Validation Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#2980b9')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#27ae60')
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.legend(loc="best")
    plt.title('Learning Curve', fontsize=14, loc='left')

    # D. Feature Importance (Handling models without checking specific attributes)
    plt.subplot(2, 2, 4)
    # Check for feature importances on the model or its sub-estimators
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'estimators_'): # For VotingRegressor, average importances if possible
        try:
            imps = [est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')]
            if imps:
                importances = np.mean(imps, axis=0)
        except:
            pass
            
    if importances is not None:
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(len(importances))]
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title('Top 10 Important Features', fontsize=14, loc='left')
    else:
        plt.text(0.5, 0.5, 'Feature Importance not available\n(Model type does not support it directly)', 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.show()


# ==========================================
# 2. RUNNABLE SCRIPT SECTION
# ==========================================
# This block allows you to run this file directly to test the curves
# It re-loads the data and the saved model, then runs the plot function.

if __name__ == "__main__":
    print("🚀 Running in STANDALONE mode...")
    print("   (To use in Notebook, copy the function above into a cell)")
    
    base_dir = os.path.dirname(__file__)
    
    # 1. Load Data
    csv_path = os.path.join(base_dir, '../car_data.csv')
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        exit(1)
        
    df = pd.read_csv(csv_path)
    
    # 2. Re-create Features (Must match training exactly)
    current_year = 2024
    df['Car_Age'] = current_year - df['Year']
    df['Price_Per_Year'] = df['Present_Price'] / (df['Car_Age'] + 1)
    df['Kms_Per_Year'] = df['Kms_Driven'] / (df['Car_Age'] + 1)
    
    # 3. Load Encoders
    try:
        with open(os.path.join(base_dir, 'car_name_encoder.pkl'), 'rb') as f: car_enc = pickle.load(f)
        with open(os.path.join(base_dir, 'fuel_type_encoder.pkl'), 'rb') as f: fuel_enc = pickle.load(f)
        with open(os.path.join(base_dir, 'seller_type_encoder.pkl'), 'rb') as f: seller_enc = pickle.load(f)
        with open(os.path.join(base_dir, 'transmission_encoder.pkl'), 'rb') as f: trans_enc = pickle.load(f)
        
        df['Car_Name_Encoded'] = car_enc.transform(df['Car_Name'])
        df['Fuel_Type_Encoded'] = fuel_enc.transform(df['Fuel_Type'])
        df['Seller_Type_Encoded'] = seller_enc.transform(df['Seller_Type'])
        df['Transmission_Encoded'] = trans_enc.transform(df['Transmission'])
    except Exception as e:
        print(f"❌ Error loading encoders: {e}")
        exit(1)

    # 4. Prepare X and y
    feature_cols = [
        'Year', 'Present_Price', 'Kms_Driven', 'Owner',
        'Car_Name_Encoded', 'Fuel_Type_Encoded', 'Seller_Type_Encoded', 'Transmission_Encoded',
        'Car_Age', 'Price_Per_Year', 'Kms_Per_Year'
    ]
    X = df[feature_cols]
    y = df['Selling_Price']
    
    # 5. Load Scaler and Transform
    try:
        with open(os.path.join(base_dir, 'scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        exit(1)
        
    # 6. Load Model
    try:
        with open(os.path.join(base_dir, 'model.pkl'), 'rb') as f: model = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

    # 7. Split Data (same seed)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("✅ Environment reproduced from saved files.")
    
    # 8. Run Plot
    plot_performance_curves(model, X_train, y_train, X_test, y_test)
