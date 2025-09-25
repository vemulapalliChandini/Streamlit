import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("📈 ML Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Select columns
    date_col = st.selectbox("Select Date Column", df.columns)
    value_col = st.selectbox("Select Value Column", df.columns)

    # Convert and rename columns
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})

    # Forecast button
    if st.button("Run Forecast"):
        model = Prophet()
        model.fit(df)

        # Future dataframe (next 30 days)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Show forecast data
        st.subheader("Forecasted Values")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

        # Plot
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
