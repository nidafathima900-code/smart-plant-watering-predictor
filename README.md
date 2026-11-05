# 🌿 Smart Plant Watering Predictor

An **AIML (Artificial Intelligence & Machine Learning) mini project** that predicts whether a plant needs watering, based on soil moisture, temperature, and humidity.

**Team:** Nida Fathima · Amulya B Jain · Harmain Khanum
**Course:** BCA, Semester 5

## About the project

The model uses a **Decision Tree Classifier** (a supervised machine learning algorithm) to learn when a plant should be watered. It works like a flowchart of yes/no questions, for example: "Is the soil moisture below 30%?"

## Dataset

The dataset is **synthetically generated** (500 samples) because live sensor data was not available. Labels are created from a realistic rule (dry soil, high temperature and low humidity increase the need for water) with random noise added so the model has to learn real patterns.

## Results

| Metric | Value |
|---|---|
| Accuracy | 88% |
| Algorithm | Decision Tree (max depth = 4) |
| Train / Test split | 80% / 20% |

## Files

| File | Description |
|---|---|
| `plant_watering_predictor.py` | Main ML code: data generation, training, evaluation, live prediction |
| `index.html` | Interactive demo using the exact rules exported from the trained tree |
| `Smart_Plant_Watering_Predictor_Report.docx` | Full project report |
| `requirements.txt` | Python libraries needed |

## How to run

1. Install Python 3 and the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Run the model:
   ```
   python plant_watering_predictor.py
   ```
3. Open `index.html` in any browser to try the live demo.

## Tech used

Python · pandas · NumPy · scikit-learn · HTML · CSS · JavaScript

## Future scope

- Connect to real IoT soil moisture and weather sensors
- Add plant type as a feature (different plants need different watering)
- Compare with Random Forest and SVM
