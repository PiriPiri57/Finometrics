# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    fetch_stock_data, add_features, split_latest_row, prepare_data,
    train_regression_models, predict_today,
    train_classification_model, predict_gain_fall,
    predict_sarima, backtest_strategy_from_model, generate_signals
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS):", value="RELIANCE.NS")

if ticker:
   
    df_raw = fetch_stock_data(ticker)
    df = add_features(df_raw.copy())
    df, today_data = split_latest_row(df_raw.copy(), df.copy())
    today_data_df = pd.DataFrame([df.iloc[-1]], columns=df.columns)

    # Train Models
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    results_df, trained_models = train_regression_models(X_train, X_test, y_train, y_test)
    best_model_name = results_df['R² Score'].idxmax()
    best_model = trained_models[best_model_name]

    best_model_class, scaler_cls, best_class_name = train_classification_model(df.copy())

    # Buttons Section
    st.subheader("🔘 Select an Action")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📉 Predict Gain/Fall"):
            pred, prob = predict_gain_fall(best_model_class, scaler_cls, today_data_df)
            outcome = "📈 Gain" if pred == 1 else "📉 Fall"
            st.metric("Prediction", outcome, f"{prob*100:.2f}% Confidence")

        if st.button("📊 Show Buy/Sell/Hold Strategy + Backtest"):
            df_signals, final_value, total_return = backtest_strategy_from_model(best_model, X_test, y_test)
            st.subheader("Backtest Portfolio Value")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_signals.index, df_signals['Portfolio_Value'], label='Portfolio Value', color='blue')
            ax.set_title("Portfolio Growth Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value (₹)")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.metric("Final Portfolio Value", f"₹{final_value:.2f}")
            st.metric("Total Strategy Return", f"{total_return:.2f}%")

            st.subheader("Signal Distribution")
            fig2, ax2 = plt.subplots()
            sns.countplot(x='Signal', data=df_signals, ax=ax2)
            ax2.set_title("Distribution of Buy/Sell/Hold Signals")
            ax2.grid(True)
            st.pyplot(fig2)

    with col2:
        if st.button("💰 Predict Close Price"):
            predicted_close, actual_close, accuracy = predict_today(
                best_model, scaler, today_data_df, today_data, best_model_name
            )
            st.metric("Predicted Close", f"₹{predicted_close:.2f}")
            st.metric("Actual Close", f"₹{actual_close:.2f}")
            st.metric("Accuracy", f"{accuracy:.2f}%")

    with col3:
        if st.button("🔍 Predict Close Using SARIMA"):
            predicted_sarima, actual_sarima, acc_sarima = predict_sarima(df_raw.copy(), today_data)
            st.metric("Predicted Close (SARIMA)", f"₹{predicted_sarima:.2f}")
            st.metric("Actual Close", f"₹{actual_sarima:.2f}")
            st.metric("Accuracy", f"{acc_sarima:.2f}%")

        if st.button("📆 Forecast Next 365 Days (SARIMA)"):
            series = df_raw['Close']
            model = SARIMAX(series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24),
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit()
            forecast = results.forecast(steps=365)
            st.line_chart(forecast)
            st.success("Displayed 365-day forecast using SARIMA")

    # Stock Data and Model Results
    st.subheader("📜 Raw Stock Data")
    st.dataframe(df_raw.tail())

   
