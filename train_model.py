"""Data clean karke Random Forest train karta hai aur model/model.pkl me save karta hai."""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv"
LOCAL = "data/cardekho_dataset.csv"
MODEL_PATH = "model/model.pkl"


def load_data():
    if os.path.exists(LOCAL):
        return pd.read_csv(LOCAL)
    df = pd.read_csv(URL)          # internet se dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv(LOCAL, index=False)
    return df


def get_clean_df():
    df = load_data()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"selling_price": "price", "fuel_type": "fuel", "vehicle_age": "car_age", "transmission_type": "transmission"})
    if df["price"].max() > 1000:
        df["price"] = df["price"] / 100000          # rupees -> Lakh
    keep = ["price", "car_age", "km_driven", "fuel", "seller_type", "transmission", "mileage", "engine", "max_power", "seats"]
    df = df[keep].dropna()
    df = df[(df["km_driven"] < 500000) & (df["price"] < 100)]
    return df.reset_index(drop=True)


def train_and_save():
    df = get_clean_df()
    df_model = pd.get_dummies(df, drop_first=True)
    X, y = df_model.drop("price", axis=1), df_model["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2, mae = r2_score(y_test, pred), mean_absolute_error(y_test, pred)
    print(f"R2 = {r2:.3f} | MAE = {mae:.2f} Lakh")

    os.makedirs("model", exist_ok=True)
    bundle = {"model": model, "columns": list(X.columns), "r2": r2, "mae": mae}
    joblib.dump(bundle, MODEL_PATH, compress=3)
    return bundle


if __name__ == "__main__":
    train_and_save()
    print("Model save ho gaya:", MODEL_PATH)
