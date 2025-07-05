# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# load the dataset
print("Loading dataset...")
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df = df.sort_values(by=['id', 'date'])

# Extract year and month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Define pollutants
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Drop missing values
df = df.dropna(subset=pollutants)

# Feature and target selection
X = df[['id', 'year']]
y = df[pollutants]

# Encoding
X_encoded = pd.get_dummies(X, columns=['id'], drop_first=True)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print("Training model...")
# Train the model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

print("Saving model and columns...")
# Save the model and columns
joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")

print("Done! Model and column structure have been saved.")
