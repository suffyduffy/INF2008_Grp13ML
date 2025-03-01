import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler

# Load dataset
data = pd.read_csv("cleanedHDB.csv")

# Convert 'month' to datetime format
data['month'] = pd.to_datetime(data['month'])
data['year'] = data['month'].dt.year

# Convert resale_price to numeric (remove "$" and ",")
data['resale_price'] = data['resale_price'].replace('[\$,]', '', regex=True).astype(float)

# Encode 'town' using LabelEncoder
label_encoder = LabelEncoder()
data['town_encoded'] = label_encoder.fit_transform(data['town'])

# Define features and target
features = ["remaining_years", "floor_area_sqm", "town_encoded"]
target = "resale_price"

# Ensure dataset only contains relevant columns
data = data[features + [target, "month", "year"]]

# Remove outliers (99th percentile)
upper_limit = data[target].quantile(0.99)
data = data[data[target] <= upper_limit]

# StandardScaler to scale numerical data
scaler = StandardScaler()
X = data.drop(columns=[target, "month"])
y = data[target]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Neural Network model with increased complexity and adjustments
model = Sequential([
    Dense(512, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train_scaled.shape[1],)),  # Increased size, reduced regularization
    Dropout(0.3),  # Adjusted dropout to prevent overfitting
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),  # Reduced L2 penalty
    Dropout(0.2),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model with a slightly lower learning rate
optimizer = Adam(learning_rate=0.0001)  # Further reduced learning rate
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)  # Increased patience
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping], verbose=1)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# Calculate R² score and MAE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Debugging: Check a few predictions vs actual
print(pd.DataFrame({"Actual": y_test[:10].values, "Predicted": y_pred[:10]}))

# Future Predictions
data['future_year'] = data['year'] + 8  # Predict 8 years into the future
data['remaining_years'] -= 8  # Adjust remaining lease years
X_future = data[features]
X_future_scaled = scaler.transform(X_future)
future_predictions = model.predict(X_future_scaled).flatten()

data['predicted_resale_price'] = future_predictions

# Calculate average predicted price per year
avg_predicted_prices = data.groupby("future_year")["predicted_resale_price"].mean()
avg_predicted_prices.index = avg_predicted_prices.index.astype(int)

# Print and plot results
print(avg_predicted_prices)

plt.figure(figsize=(8, 4))
plt.plot(avg_predicted_prices.index, avg_predicted_prices.values, linestyle="-", color="blue", label="Avg Predicted Resale Price")
plt.xticks(avg_predicted_prices.index)
plt.xlabel("Upcoming Years")
plt.ylabel("Average Predicted Resale Price ($)")
plt.title("Resale Prediction by Neural Network")
plt.legend()
plt.grid(True)
plt.show()
