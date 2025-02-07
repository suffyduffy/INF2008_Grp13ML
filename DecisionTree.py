from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("cleanedHDB.csv")
# Convert 'month' to datetime format for proper plotting
df['month'] = pd.to_datetime(df['month'])

# Encode categorical 'street_name' into numbers
#label_encoder = LabelEncoder()
#df['street_name'] = label_encoder.fit_transform(df['street_name'])

if 'resale_price' in df.columns:
    print("Column exists")
else:
    print("Column does not exist")

# according to my research(totally chatgpt), this 2 affects the resale value the most
features = ["remaining_years", "floor_area_sqm"]
target = "resale_price"

# Ensure selected columns exist in the dataset
df = df[features + [target]]

# Split the data into training and testing sets
X = df.drop(columns=[target])
y = df[target]

# Ensure the new data contains only the required feature columns
new_data = df[["remaining_years", "floor_area_sqm"]]

####################################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model on the scaled data
reg = DecisionTreeRegressor(max_depth=5, random_state=42)
reg.fit(X_train_scaled, y_train)

# Scale the new data using the previously fitted scaler
new_data_scaled = scaler.transform(new_data)

# Predict resale prices for the new data
future_predictions = reg.predict(new_data_scaled)

# Scale the new data using the same scaler fitted on the training data
new_data_scaled = scaler.transform(new_data)

# Predict resale prices for the new data
future_predictions = reg.predict(new_data_scaled)

# Display predicted values
print("Predicted resale prices for future data:", future_predictions)


