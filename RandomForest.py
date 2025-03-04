from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("cleanedHDB.csv")

# Convert 'month' to datetime format for proper plotting
df['month'] = pd.to_datetime(df['month'])

df[["storey_min", "storey_max"]] = df["storey_range"].str.split(" TO ", expand=True).astype(int)
# Compute average storey
df["storey_avg"] = (df["storey_min"] + df["storey_max"]) / 2

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the 'town' column
df['town_encoded'] = label_encoder.fit_transform(df['town'])
df['estateType_encoded'] = label_encoder.fit_transform(df['estate_type'])

if 'resale_price' in df.columns:
    print("Column exists")
else:
    print("Column does not exist")

# Features that affect resale value based on research on goo0gle
features = ["remaining_years", "floor_area_sqm", "town_encoded", "estateType_encoded", "storey_avg"]
target = "resale_price"

# Ensure the dataset contains only the selected columns
df = df[features + [target] + ['month']]

# Remove outliers by filtering resale prices beyond the 99th percentile
upper_limit = df['resale_price'].quantile(0.99)
df = df[df['resale_price'] <= upper_limit]

# Prepare feature matrix (X) and target vector (y)
X = df.drop(columns=[target, 'month'])
y = df[target]

# Select only the features for future predictions
new_data = df[features]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#StandardScaler to scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

# Train the model with the scaled training data
reg.fit(X_train_scaled, y_train)
y_predict = reg.predict(X_test_scaled)

# performance metric
r2 = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)


# Scale the new data for future predictions using the trained scaler
new_data_scaled = scaler.transform(new_data)
future_predictions = reg.predict(new_data_scaled)

df['predicted_resale_price'] = future_predictions
df['year'] = df['month'].dt.year
df['future_year'] = df['year'] + 8  # Set future year for plotting purposes
# Calculate the average predicted resale price per future year
avg_predicted_prices = df.groupby('future_year')['predicted_resale_price'].mean()
avg_predicted_prices.index = avg_predicted_prices.index.astype(int)

# Print the average predicted prices per year and performance metrics
print(avg_predicted_prices)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Plot the average predicted resale price over the upcoming years
plt.figure(figsize=(8, 4))
plt.plot(avg_predicted_prices.index, avg_predicted_prices.values,  linestyle='-', color='green',
         label='Avg Predicted Resale Price')
plt.xticks(avg_predicted_prices.index)
plt.xlabel("Upcoming Years")
plt.ylabel("Average Predicted Resale Price ($)")
plt.title("Resale Prediction by Decision Tree")
plt.legend()
plt.grid(True)
plt.show()
