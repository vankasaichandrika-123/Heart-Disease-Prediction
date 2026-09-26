from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open("Heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaled_model.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            features = [
                float(request.form["age_yeo_trim"]),
                float(request.form["sex_yeo_trim"]),
                float(request.form["cp_yeo_trim"]),
                float(request.form["thalach_yeo_trim"]),
                float(request.form["oldpeak_yeo_trim"]),
                float(request.form["slope_yeo_trim"]),
                float(request.form["thal_yeo_trim"])

            ]

            # Convert to numpy array
            features_array = np.array([features])

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Make prediction
            prediction = model.predict(features_scaled)[0]
            if prediction == 0:
                return render_template("index.html", prediction='patient fine')
            else:
                return render_template("index.html", prediction="patient not fine")

        except Exception as e:
            # If an error occurs during POST, render the template with the error message
            prediction = f"Error: {str(e)}"
            return render_template("index.html", prediction=prediction) # <--- Added return here for error handling

    # This handles the initial GET request (when the page is loaded)
    # and any scenario where the POST block didn't execute or encountered an error.
    return render_template("index.html", prediction=prediction) # <--- ADDED REQUIRED RETURN STATEMENT

if __name__ == "__main__":
    app.run(debug=True)