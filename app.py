from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# =============================
# LOAD MODELS
# =============================
MODEL_PATH = "model/"

stack_model = joblib.load(MODEL_PATH + "stack_model.pkl")
scaler = joblib.load(MODEL_PATH + "scaler.pkl")
features = joblib.load(MODEL_PATH + "features.pkl")

# =============================
# VALID INPUT RANGES
# =============================
RANGES = {
    "engine": (0.6, 8.0),
    "cylinders": (3, 12),
    "city": (1, 30),
    "highway": (1, 25),
    "combined": (1, 28)
}

# =============================
# ROUTES
# =============================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict-page")
def predict_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        engine = float(request.form["engine"])
        cylinders = int(request.form["cylinders"])
        transmission = request.form["transmission"]
        fuel = request.form["fuel"]
        city = float(request.form["city"])
        highway = float(request.form["highway"])
        combined = float(request.form["combined"])

        inputs = {
            "engine": engine,
            "cylinders": cylinders,
            "city": city,
            "highway": highway,
            "combined": combined
        }

        for key, value in inputs.items():
            low, high = RANGES[key]

            if not (low <= value <= high):
                return render_template(
                    "error.html",
                    message=f"Invalid value for {key}. Allowed range: {low} – {high}"
                )

        # =============================
        # CREATE INPUT DATAFRAME
        # =============================
        input_df = pd.DataFrame(columns=features)
        input_df.loc[0] = 0

        input_df["Engine Size(L)"] = engine
        input_df["Cylinders"] = cylinders
        input_df["Fuel Consumption City (L/100 km)"] = city
        input_df["Fuel Consumption Hwy (L/100 km)"] = highway
        input_df["Fuel Consumption Comb (L/100 km)"] = combined

        input_df["City_to_Hwy_Ratio"] = city / highway
        input_df["Engine_per_Cylinder"] = engine / cylinders

        # =============================
        # ONE HOT ENCODING
        # =============================
        if f"Transmission_{transmission}" in input_df.columns:
            input_df[f"Transmission_{transmission}"] = 1

        if f"Fuel Type_{fuel}" in input_df.columns:
            input_df[f"Fuel Type_{fuel}"] = 1

        # =============================
        # SCALE
        # =============================
        X_scaled = scaler.transform(input_df)

        prediction = round(stack_model.predict(X_scaled)[0], 2)

        # =============================
        # CATEGORY
        # =============================
        if prediction < 150:
            category = "Low Emission"
            color = "green"

        elif prediction < 250:
            category = "Moderate Emission"
            color = "orange"

        else:
            category = "High Emission"
            color = "red"

        # =============================
        # SUGGESTIONS
        # =============================
        suggestions = []

        if engine > 3:
            suggestions.append("Reduce engine size to lower CO₂ emissions.")

        if cylinders > 6:
            suggestions.append("Vehicles with fewer cylinders emit less CO₂.")

        if combined > 10:
            suggestions.append("Improve fuel efficiency through smooth driving.")

        if fuel.lower() == "diesel":
            suggestions.append("Consider hybrid or electric vehicles.")
            
        if not suggestions:
            suggestions.append("Vehicle configuration is already optimized for low emissions.")

        # =============================
        # INPUT VALUES FOR DISPLAY
        # =============================
        input_display = {
            "Engine Size (L)": engine,
            "Cylinders": cylinders,
            "Transmission": transmission,
            "Fuel Type": fuel,
            "City Consumption (L/100 km)": city,
            "Highway Consumption (L/100 km)": highway,
            "Combined Consumption (L/100 km)": combined
        }

        return render_template(
            "result.html",
            prediction=prediction,
            category=category,
            color=color,
            suggestions=suggestions,
            inputs=input_display
        )

    except Exception as e:
        return render_template("error.html", message=str(e))


if __name__ == "__main__":
    app.run(debug=True)