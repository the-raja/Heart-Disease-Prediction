from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# --------------------------------------------------
# Load trained models dictionary
# --------------------------------------------------
models = pickle.load(open("heart_models.pkl", "rb"))

# --------------------------------------------------
# EXACT feature order used during training
# MUST match notebook FEATURE_COLUMNS
# --------------------------------------------------
FEATURE_ORDER = [
    'age',
    'sex',
    'dataset',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalch',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal'
]

# --------------------------------------------------
# Severity mapping
# --------------------------------------------------
SEVERITY_MAP = {
    0: "No Heart Disease",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Critical"
}

# --------------------------------------------------
# Home Route
# --------------------------------------------------
@app.route('/')
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction Route
# --------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:

        # ------------------------------
        # Collect input data properly
        # ------------------------------
        input_dict = {}

        for feature in FEATURE_ORDER:
            value = request.form.get(feature)

            # Convert numeric fields
            if feature in ['age', 'trestbps', 'chol', 'thalch', 'ca']:
                value = int(value)
            elif feature in ['oldpeak']:
                value = float(value)

            input_dict[feature] = value

        # Convert to DataFrame (VERY IMPORTANT)
        input_df = pd.DataFrame([input_dict])

        results = {}

        best_model_name = None
        highest_confidence = -1

        # ------------------------------
        # Predict using all models
        # ------------------------------
        for name, model in models.items():

            prediction = model.predict(input_df)[0]

            # Get probability if available
            if hasattr(model.named_steps['model'], "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                confidence = round(max(probabilities) * 100, 2)
            else:
                confidence = 0

            results[name] = {
                "prediction": SEVERITY_MAP.get(prediction, "Unknown"),
                "confidence": confidence
            }

            # Detect best model (highest confidence)
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_model_name = name

        return render_template(
            "result.html",
            results=results,
            best_model=best_model_name,
            best_confidence=highest_confidence
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


# --------------------------------------------------
# Run Application
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)