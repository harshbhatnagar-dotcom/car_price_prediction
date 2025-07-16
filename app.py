from flask import Flask,render_template,request
import joblib
import pandas as pd


app=Flask(__name__)
df=pd.read_csv("cardekho_dataset.csv")
model=joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html",
        brands=df["brand"].unique(),
        models=df["model"].unique(),
        seller_types=df["seller_type"].unique(),
        fuel_types=df["fuel_type"].unique(),
        transmission_types=df["transmission_type"].unique()
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "brand": request.form["brand"],
            "model": request.form["model"],
            "vehicle_age": int(request.form["vehicle_age"]),
            "km_driven": int(request.form["km_driven"]),
            "seller_type": request.form["seller_type"],
            "fuel_type": request.form["fuel_type"],
            "transmission_type": request.form["transmission_type"],
            "mileage": float(request.form["mileage"]),
            "engine": float(request.form["engine"]),
            "max_power": float(request.form["max_power"]),
            "seats": int(request.form["seats"])
        }

        df_input = pd.DataFrame([input_data])
        predicted_price = int(round(model.predict(df_input)[0], -3))

        return render_template("prediction.html", data=input_data, price=predicted_price)

    except Exception as e:
        return f"Prediction Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)





if __name__=="__main__":
    app.run(debug=True)
