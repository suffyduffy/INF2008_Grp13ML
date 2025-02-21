import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("cleanedHDB.csv")

# Clean column names (strip spaces)
data.columns = data.columns.str.strip()

# Convert 'month' to datetime format and extract 'year'
data['month'] = pd.to_datetime(data['month'])
data['year'] = data['month'].dt.year

# Convert resale_price to numeric (remove "$" and ",")
data['resale_price'] = data['resale_price'].replace('[\$,]', '', regex=True).astype(float)

# Define features and target
features = ["remaining_years", "floor_area_sqm", "year", "flat_type", "storey_range", "flat_model", "town"]
target = "resale_price"

# Ensure dataset only contains relevant columns
data = data[features + [target]]

# Remove outliers (99th percentile)
upper_limit = data[target].quantile(0.99)
data = data[data[target] <= upper_limit]

# Handle categorical features using OneHotEncoding
categorical_features = ["flat_type", "storey_range", "flat_model", "town"]
numerical_features = ["remaining_years", "floor_area_sqm", "year"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)  # FIX: Use sparse_output=False
    ]
)

# Prepare feature matrix (X) and target vector (y)
X = data.drop(columns=[target])
y = data[target]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform data
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Define Neural Network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Output layer (Regression problem)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Future Predictions (now considering year, flat type, storey, and town)
future_data = data.copy()
future_data["year"] += 8  # Predict 8 years into the future
X_future = future_data.drop(columns=[target])

# Scale future data
X_future_scaled = preprocessor.transform(X_future)
future_predictions = model.predict(X_future_scaled).flatten()

# Store predictions
future_data["predicted_resale_price"] = future_predictions

# Calculate average predicted price per year
avg_predicted_prices = future_data.groupby("year")["predicted_resale_price"].mean()
avg_predicted_prices.index = avg_predicted_prices.index.astype(int)

# Print and plot results
print(avg_predicted_prices)

plt.figure(figsize=(8, 4))
plt.plot(avg_predicted_prices.index, avg_predicted_prices.values, linestyle="-", color="blue", label="Avg Predicted Resale Price")
plt.xticks(avg_predicted_prices.index)
plt.xlabel("Upcoming Years")
plt.ylabel("Average Predicted Resale Price ($)")
plt.title("Resale Prediction by Neural Network (with Flat Type, Storey, Model, Town)")
plt.legend()
plt.grid(True)
plt.show()
