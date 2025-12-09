import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚗 ADVANCED CAR PRICE PREDICTION MODEL - TRAINING")
print("=" * 70)

def plot_correlation_heatmap(data):
    """Affiche la matrice de corrélation des features numériques."""
    print("\n📊 Generating Correlation Heatmap...")
    plt.figure(figsize=(14, 12))
    
    # Sélectionner uniquement les colonnes numériques
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculer la corrélation
    correlation = numeric_data.corr()
    
    # Afficher la Heatmap
    sns.heatmap(correlation, annot=True, cmap='RdBu', fmt=".2f", linewidths=0.5)
    
    plt.title('Matrice de Corrélation des Features', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Afficher spécifiquement la corrélation avec le Prix de Vente
    print("   ✓ Correlation with Selling_Price:")
    target_corr = correlation['Selling_Price'].sort_values(ascending=False)
    print(target_corr.head(10))

def plot_performance_curves(model, X_train, y_train, X_test, y_test):
    """Affiche les courbes de performance du modèle."""
    print("\n📊 Generating Performance Curves...")
    
    # Prédire sur le jeu de test
    y_pred = model.predict(X_test)
    
    # Configuration de la figure
    plt.figure(figsize=(20, 15))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Graphique Réel vs Prédit
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='#2c3e50')
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Prix Réel (y_test)', fontsize=12)
    plt.ylabel('Prix Prédit (y_pred)', fontsize=12)
    plt.title('Performance: Réel vs Prédit', fontsize=14, loc='left')
    
    # 2. Distribution des Résidus
    residuals = y_test - y_pred
    plt.subplot(2, 2, 2)
    sns.histplot(residuals, kde=True, color='#e74c3c')
    plt.axvline(x=0, color='k', linestyle='--', lw=1)
    plt.xlabel('Erreur (Réel - Prédit)', fontsize=12)
    plt.title('Distribution des Erreurs (Résidus)', fontsize=14, loc='left')

    # 3. Courbe d'Apprentissage
    plt.subplot(2, 2, 3)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color='#2980b9', label='Score Entraînement')
    plt.plot(train_sizes, val_mean, 'o-', color='#27ae60', label='Score Validation (Cross-Val)')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#2980b9')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#27ae60')
    plt.xlabel('Taille du jeu d\'entraînement', fontsize=12)
    plt.ylabel('Score R²', fontsize=12)
    plt.legend(loc="best")
    plt.title('Courbe d\'Apprentissage (Bias vs Variance)', fontsize=14, loc='left')

    # 4. Importance des Features
    plt.subplot(2, 2, 4)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(len(importances))]
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title('Top 10 Caractéristiques Importantes', fontsize=14, loc='left')
    else:
        plt.text(0.5, 0.5, 'Pas d\'importance des features\ndisponible pour ce modèle', 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.show()

# Load the actual CSV data
csv_path = os.path.join(os.path.dirname(__file__), '../car_data.csv')
print(f"\n📂 Loading data from: {csv_path}")
data = pd.read_csv(csv_path)
print(f"✓ Loaded {len(data)} records")

# Display basic info
print(f"\n📊 Dataset Overview:")
print(f"   Columns: {list(data.columns)}")
print(f"   Price range: ${data['Selling_Price'].min():.2f} - ${data['Selling_Price'].max():.2f}")
print(f"   Year range: {data['Year'].min()} - {data['Year'].max()}")

# ============================================
# FEATURE ENGINEERING (Advanced)
# ============================================
print("\n🔧 Feature Engineering...")

# Create new features
current_year = 2024
data['Car_Age'] = current_year - data['Year']
data['Price_Per_Year'] = data['Present_Price'] / (data['Car_Age'] + 1)
data['Kms_Per_Year'] = data['Kms_Driven'] / (data['Car_Age'] + 1)
data['Depreciation'] = data['Present_Price'] - data['Selling_Price']
data['Depreciation_Rate'] = (data['Depreciation'] / data['Present_Price']) * 100

print("✓ Created advanced features:")
print("   - Car_Age (current year - manufacturing year)")
print("   - Price_Per_Year (price depreciation per year)")
print("   - Kms_Per_Year (average annual mileage)")
print("   - Depreciation (absolute price loss)")
print("   - Depreciation_Rate (percentage price loss)")

# Encoding categorical variables
print("\n🔤 Encoding categorical variables...")
fuel_type_encoder = LabelEncoder()
seller_type_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
car_name_encoder = LabelEncoder()

data['Fuel_Type_Encoded'] = fuel_type_encoder.fit_transform(data['Fuel_Type'])
data['Seller_Type_Encoded'] = seller_type_encoder.fit_transform(data['Seller_Type'])
data['Transmission_Encoded'] = transmission_encoder.fit_transform(data['Transmission'])
data['Car_Name_Encoded'] = car_name_encoder.fit_transform(data['Car_Name'])

print(f"✓ Fuel types: {list(fuel_type_encoder.classes_)}")
print(f"✓ Seller types: {list(seller_type_encoder.classes_)}")
print(f"✓ Transmissions: {list(transmission_encoder.classes_)}")
print(f"✓ Unique cars: {len(car_name_encoder.classes_)} models")

# Visualize Correlation
plot_correlation_heatmap(data)

# Select features and target (with engineered features)
feature_columns = [
    'Year', 'Present_Price', 'Kms_Driven', 'Owner',
    'Car_Name_Encoded', 'Fuel_Type_Encoded', 
    'Seller_Type_Encoded', 'Transmission_Encoded',
    'Car_Age', 'Price_Per_Year', 'Kms_Per_Year'
]

X = data[feature_columns]
y = data['Selling_Price']

# Feature Scaling for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data Split:")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")

# ============================================
# MODEL TRAINING (Multiple Advanced Models)
# ============================================
print("\n" + "=" * 70)
print("🤖 TRAINING ADVANCED MODELS")
print("=" * 70)

# Model 1: Optimized Random Forest
print("\n1️⃣ Training Optimized Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_train_score = rf_model.score(X_train, y_train)
rf_test_score = rf_model.score(X_test, y_test)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"   ✓ Random Forest Results:")
print(f"      R² (Train): {rf_train_score:.4f}")
print(f"      R² (Test):  {rf_test_score:.4f}")
print(f"      MAE:        ${rf_mae:.2f}")
print(f"      RMSE:       ${rf_rmse:.2f}")

# Model 2: Gradient Boosting
print("\n2️⃣ Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_train_score = gb_model.score(X_train, y_train)
gb_test_score = gb_model.score(X_test, y_test)
gb_pred = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print(f"   ✓ Gradient Boosting Results:")
print(f"      R² (Train): {gb_train_score:.4f}")
print(f"      R² (Test):  {gb_test_score:.4f}")
print(f"      MAE:        ${gb_mae:.2f}")
print(f"      RMSE:       ${gb_rmse:.2f}")

# Model 3: Ensemble Voting Regressor (Best of both)
print("\n3️⃣ Creating Ensemble Model (Voting)...")
ensemble_model = VotingRegressor([
    ('rf', rf_model),
    ('gb', gb_model)
])
ensemble_model.fit(X_train, y_train)
ensemble_train_score = ensemble_model.score(X_train, y_train)
ensemble_test_score = ensemble_model.score(X_test, y_test)
ensemble_pred = ensemble_model.predict(X_test)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print(f"   ✓ Ensemble Model Results:")
print(f"      R² (Train): {ensemble_train_score:.4f}")
print(f"      R² (Test):  {ensemble_test_score:.4f}")
print(f"      MAE:        ${ensemble_mae:.2f}")
print(f"      RMSE:       ${ensemble_rmse:.2f}")

# Select best model
models = {
    'Random Forest': (rf_model, rf_test_score, rf_mae),
    'Gradient Boosting': (gb_model, gb_test_score, gb_mae),
    'Ensemble': (ensemble_model, ensemble_test_score, ensemble_mae)
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model, best_score, best_mae = models[best_model_name]

print("\n" + "=" * 70)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_score:.4f} ({best_score*100:.2f}% accuracy)")
print(f"   MAE: ${best_mae:.2f}")
print("=" * 70)

# Cross-validation for robustness
print(f"\n🔄 Cross-Validation (5-fold) on {best_model_name}...")
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, 
                           scoring='r2', n_jobs=-1)
print(f"   CV Scores: {cv_scores}")
print(f"   Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance
print(f"\n📊 Top 5 Most Important Features:")
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    

    for idx, row in feature_imp.head(5).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# Visualize Performance Curves
plot_performance_curves(best_model, X_train, y_train, X_test, y_test)

# ============================================
# SAVE MODEL AND PREPROCESSORS
# ============================================
print("\n💾 Saving model and preprocessors...")

# Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save encoders
with open('fuel_type_encoder.pkl', 'wb') as f:
    pickle.dump(fuel_type_encoder, f)

with open('seller_type_encoder.pkl', 'wb') as f:
    pickle.dump(seller_type_encoder, f)

with open('transmission_encoder.pkl', 'wb') as f:
    pickle.dump(transmission_encoder, f)

with open('car_name_encoder.pkl', 'wb') as f:
    pickle.dump(car_name_encoder, f)

print("✓ All files saved successfully!")
print("\n📁 Saved files:")
print("   ✓ model.pkl (Best model: {})".format(best_model_name))
print("   ✓ scaler.pkl (Feature scaler)")
print("   ✓ fuel_type_encoder.pkl")
print("   ✓ seller_type_encoder.pkl")
print("   ✓ transmission_encoder.pkl")
print("   ✓ car_name_encoder.pkl")

print("\n" + "=" * 70)
print("✅ ADVANCED MODEL TRAINING COMPLETE!")
print(f"   Final Model: {best_model_name}")
print(f"   Accuracy: {best_score*100:.2f}%")
print(f"   Average Error: ${best_mae:.2f}")
print("=" * 70)
