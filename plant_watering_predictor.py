"""
Smart Plant Watering Predictor
Algorithm: Decision Tree Classifier
Dataset: Synthetically generated (soil moisture, temperature, humidity)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. Generate synthetic dataset
# -------------------------------
np.random.seed(42)
n_samples = 500

soil_moisture = np.random.uniform(5, 80, n_samples)      # percent
temperature = np.random.uniform(15, 45, n_samples)       # Celsius
humidity = np.random.uniform(20, 90, n_samples)          # percent

# Rule used to label data: plant needs water if soil is dry,
# and heat/low humidity make it worse. Some randomness added
# so the model doesn't just memorize a straight line.
needs_water = []
for sm, temp, hum in zip(soil_moisture, temperature, humidity):
    score = (30 - sm) + (temp - 25) * 0.5 - (hum - 50) * 0.2
    score += np.random.normal(0, 5)  # noise
    needs_water.append(1 if score > 0 else 0)

df = pd.DataFrame({
    "soil_moisture": soil_moisture,
    "temperature": temperature,
    "humidity": humidity,
    "needs_water": needs_water
})

print("Dataset generated successfully.")
print(df.head())
print("\nClass distribution:")
print(df["needs_water"].value_counts())

# -------------------------------
# 2. Prepare the data
# -------------------------------
X = df[["soil_moisture", "temperature", "humidity"]]
y = df["needs_water"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Train the model
# -------------------------------
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Water", "Water"]))

# Show the decision tree as readable rules (great for your viva)
print("\nDecision Tree Rules:")
print(export_text(model, feature_names=["soil_moisture", "temperature", "humidity"]))

# -------------------------------
# 5. Try it on your own values
# -------------------------------
def predict_watering(soil, temp, hum):
    data = pd.DataFrame(
        [[soil, temp, hum]],
        columns=["soil_moisture", "temperature", "humidity"]
    )
    prediction = model.predict(data)[0]
    return "Water the plant" if prediction == 1 else "No need to water"


print("\n--- Try it yourself ---")
samples = [
    (15, 38, 30),  # dry soil, hot, low humidity -> likely water
    (70, 20, 80),  # moist soil, cool, humid -> likely no water
    (35, 28, 50),  # borderline case
]

for soil, temp, hum in samples:
    result = predict_watering(soil, temp, hum)
    print(f"Soil={soil}%, Temp={temp}C, Humidity={hum}% -> {result}")

while True:
    soil = float(input("Soil moisture (%) or -1 to quit: "))
    if soil == -1:
        break
    temp = float(input("Temperature (C): "))
    hum = float(input("Humidity (%): "))
    print("Result:", predict_watering(soil, temp, hum))