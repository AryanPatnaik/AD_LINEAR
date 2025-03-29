from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("student_model.pkl")
    if not hasattr(model, "predict"):
        raise ValueError("Loaded model is not valid! It does not have a 'predict' method.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature names matching the dataset
FEATURES = ["Hours Studied", "Previous Scores", "Extracurricular Activities", 
            "Sleep Hours", "Sample Question Papers Practiced"]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            data = request.form

            # Ensure all required fields are provided
            if not all(feature in data for feature in FEATURES):
                return render_template('index.html', prediction="Error: Missing input values.")

            # Convert form data into a DataFrame
            input_data = pd.DataFrame([[float(data[feature]) for feature in FEATURES]], columns=FEATURES)

            # Make prediction
            prediction = round(model.predict(input_data)[0], 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    if model:
        app.run(debug=True)
    else:
        print("Model failed to load. Exiting.")
