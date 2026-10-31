import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource(show_spinner="Pehli baar model train ho raha hai, 1-2 minute lagenge...")
def load_model():
    if not os.path.exists("model/model.pkl"):
        from train_model import train_and_save
        return train_and_save()
    return joblib.load("model/model.pkl")


@st.cache_data(show_spinner=False)
def load_clean_data():
    from train_model import get_clean_df
    return get_clean_df()


try:
    bundle = load_model()
    df = load_clean_data()
except Exception as e:
    st.error(f"Model ya dataset load nahi hua: {e}")
    st.stop()

model, columns, mae, r2 = bundle["model"], bundle["columns"], bundle["mae"], bundle["r2"]

st.markdown("<h1 style='text-align:center;color:#1E3A8A;'>🚗 AI-Powered Used Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;opacity:0.7;'>End-to-End Machine Learning Web Application trained on {len(df):,} Indian used car listings (CarDekho)</p>", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Price Predictor", "📈 Dataset Overview", "⚙️ Model Insights"])

with tab1:
    st.header("🚘 Enter Car Details")
    with st.container(border=True):
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
        km_driven = st.number_input("Kilometres Driven", 0, 500000, 40000, step=1000)
        engine = st.number_input("Engine (cc)", 600, 5000, 1200, step=100)
        max_power = st.number_input("Max Power (bhp)", 30.0, 600.0, 85.0)
        mileage = st.number_input("Mileage (kmpl)", 4.0, 40.0, 18.0)
        seats = st.slider("Seats", 2, 9, 5)
        car_age = st.slider("Car Age (Years)", 0, 25, 5)
        go = st.button("💡 Estimate Car Price", use_container_width=True)

    st.header("🏷️ Valuation Estimate")
    if go:
        row = pd.DataFrame([{"car_age": car_age, "km_driven": km_driven, "fuel": fuel, "seller_type": seller_type,
                             "transmission": transmission, "mileage": mileage, "engine": engine,
                             "max_power": max_power, "seats": seats}])
        row = pd.get_dummies(row).reindex(columns=columns, fill_value=0)
        price = float(model.predict(row)[0])
        st.success(f"Estimated price: ₹{price:.2f} Lakh")
        st.info(f"Expected range: ₹{max(price - mae, 0):.2f} to ₹{price + mae:.2f} Lakh")
    else:
        st.info("Car ki details bharo aur **'Estimate Car Price'** dabao.")

with tab2:
    st.header("📈 Dataset Overview")
    a, b = st.columns(2)
    a.metric("Total cars", f"{len(df):,}")
    b.metric("Average price", f"₹{df['price'].mean():.2f} Lakh")
    a.metric("Average age", f"{df['car_age'].mean():.1f} years")
    b.metric("Average km driven", f"{df['km_driven'].mean():,.0f}")
    st.subheader("Sample data")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Price distribution (Lakh)")
    counts, bins = np.histogram(df["price"], bins=25)
    st.bar_chart(pd.Series(counts, index=np.round(bins[:-1], 1)))
    st.subheader("Car age vs price")
    st.scatter_chart(df.sample(min(2000, len(df)), random_state=1), x="car_age", y="price")
    st.subheader("Average price by fuel type (Lakh)")
    st.bar_chart(df.groupby("fuel")["price"].mean())

with tab3:
    st.header("⚙️ Model Insights")
    res = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
        "R² score": [0.665, 0.913, 0.945],
        "MAE (Lakh)": [2.56, 1.22, 0.99],
    })
    st.table(res)
    st.caption("Random Forest sabse accha raha, isliye app me wahi use hua hai.")
    st.subheader("R² score comparison")
    st.bar_chart(res.set_index("Model")["R² score"])
    st.subheader("Price par sabse zyada asar kis cheez ka hai?")
    imp = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False).head(8)
    st.bar_chart(imp)
    c1, c2 = st.columns(2)
    c1.metric("R² (this model)", f"{r2:.3f}")
    c2.metric("Average error", f"₹{mae:.2f} Lakh")

st.divider()
st.caption("Final year B.Tech ML project | Python, Pandas, Scikit-learn, Streamlit")
