# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import time
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX

def get_last_trading_day():
    today = datetime.datetime.today()
    nse = mcal.get_calendar("NSE")
    start = today - datetime.timedelta(days=10)
    end = today

    schedule = nse.schedule(
        start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d")
    )
    trading_days = schedule.index

    if today.date() in trading_days.date:
        return today.date()
    else:
        return trading_days[trading_days < np.datetime64(today)].max().date()

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    last_day = get_last_trading_day()

    for attempt in range(3):
        try:
            df = yf.download(
                ticker, end=(last_day + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            )
            if df is not None and df.empty:
                df.reset_index(inplace=True)
                df.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors="ignore")
                return df
        except Exception :
            time.sleep(3)

  
    try:
        fallback_df = pd.read_csv("DS_Dataset_FinanceTrends.csv")
        fallback_df["Date"] = pd.to_datetime(fallback_df["Date"],errors="coerce")
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        fallback_df[numeric_cols] = fallback_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        fallback_df.dropna(inplace=True)  # ✅ Drop invalid rows
        st.warning("Live data fetch failed. Using fallback data from DS_Dataset_FinanceTrends.csv.")
        return fallback_df
    except FileNotFoundError:
        raise ValueError("Both live fetch and fallback CSV failed. Please ensure reliance.csv is available.")

def add_features(df):
    """
    Adds time-based and technical indicators to the stock dataframe.
    Args:
        df (pd.DataFrame): Raw stock data
    Returns:
        pd.DataFrame: DataFrame with added features
    """
#    # 🔍 Inspect raw input
#     st.write("🔍 dtypes BEFORE conversion:", df.dtypes)
#     st.write("🔍 First few rows:", df.head())

#     # ✅ Convert important columns to numeric (forcefully)
#     numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
#     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

#     # ✅ Convert Date column to datetime if present
#     if "Date" in df.columns:
#         df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

#     # ✅ Drop rows with any missing values
#     df.dropna(inplace=True)

#     # 🔍 Confirm cleanup
#     st.write("✅ dtypes AFTER cleaning:", df.dtypes)

    # 🧠 Feature engineering
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    df["lag_1"] = df["Close"].shift(1)
    df["lag_5"] = df["Close"].shift(5)
    df["lag_10"] = df["Close"].shift(10)

    df["rolling_mean_5"] = df["Close"].rolling(window=5).mean()
    df["rolling_mean_10"] = df["Close"].rolling(window=10).mean()
    df["rolling_mean_20"] = df["Close"].rolling(window=20).mean()

    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["Log_Volume"] = df["Volume"].shift(1)
    df["Rolling_Volume"] = df["Volume"].rolling(3).mean()

    # Final cleanup
    df.dropna(inplace=True)
    df.drop(
        columns=[
            "Date",
            "lag_5",
            "lag_10",
            "Open",
            "Low",
            "Rolling_Volume",
            "Daily_Return",
            "IsMonthStart",
            "rolling_mean_10",
            "rolling_mean_20",
        ],
        inplace=True,
        errors="ignore"
    )

    # 🧾 Final feature selection
    df = df[['High', 'Close', 'Volume', 'lag_1', 'rolling_mean_5', 'Log_Return']]
    return df

def split_latest_row(df_raw, df):
    today_data = df_raw.iloc[-1]
    df = df.iloc[:-1]
    if df_raw.empty or df.empty:
        raise ValueError("DataFrame is empty. Cannot split latest row.")

    return df, today_data

def prepare_data(df):
    df = df.sort_index()
    X = df.drop(['Close'], axis=1)
    y = df['Close']
    
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_regression_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R² Score': r2
        }
        trained_models[name] = model

    results_df = pd.DataFrame(results).T
    return results_df, trained_models
def predict_today(best_model, scaler, today_data_df, today_data, best_model_name):
    """
    Predict today's closing price using the trained model and display the results.
    
    Returns:
        predicted_close (float): Predicted closing price
        actual_close (float): Actual closing price
        accuracy (float): Percentage accuracy
    """
    today_data_df.fillna(0, inplace=True)

    # Select only the features used in model training
    today_data_df = today_data_df[['High', 'Volume', 'lag_1', 'rolling_mean_5', 'Log_Return']]
    today_data_scaled = scaler.transform(today_data_df)

    today_prediction = best_model.predict(today_data_scaled)
    predicted_close = today_prediction.flatten()[0]

    actual_close = today_data['Close'].item()

    accuracy = (1 - abs(actual_close - predicted_close) / actual_close) * 100

    return predicted_close, actual_close, accuracy
def train_classification_model(df):
    df['Gain/Fall'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)

    X_cls = df.drop(['Close', 'Gain/Fall'], axis=1)
    y_cls = df['Gain/Fall']

    train_size_cls = int(len(df) * 0.8)
    X_train_cls, X_test_cls = X_cls.iloc[:train_size_cls], X_cls.iloc[train_size_cls:]
    y_train_cls, y_test_cls = y_cls.iloc[:train_size_cls], y_cls.iloc[train_size_cls:]

    scaler_cls = StandardScaler()
    X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler_cls.transform(X_test_cls)

    models_class = [
        LogisticRegression(),
        XGBClassifier(),
        RandomForestClassifier(n_estimators=100, random_state=42),
        DecisionTreeClassifier(random_state=42),
        KNeighborsClassifier(n_neighbors=5),
        GradientBoostingClassifier(n_estimators=100, random_state=42)
    ]

    validation_scores = {}

    for model in models_class:
        model.fit(X_train_cls_scaled, y_train_cls)
        score = roc_auc_score(y_test_cls, model.predict_proba(X_test_cls_scaled)[:, 1])
        validation_scores[model.__class__.__name__] = score

    best_model_class_name = max(validation_scores, key=validation_scores.get)
    best_model_class = models_class[[model.__class__.__name__ for model in models_class].index(best_model_class_name)]

    return best_model_class, scaler_cls, best_model_class_name


def predict_gain_fall(clf, scaler_cls, today_data_df):
    today_data_df.fillna(0, inplace=True)
    today_data_df = today_data_df[['High', 'Volume', 'lag_1', 'rolling_mean_5', 'Log_Return']]
    today_data_scaled = scaler_cls.transform(today_data_df)
    prediction = clf.predict(today_data_scaled)[0]
    probability = clf.predict_proba(today_data_scaled)[0][prediction]
    return prediction, probability
def predict_sarima(df_raw, today_data):
    series = df_raw['Close']
    order = (1, 0, 1)
    seasonal_order = (1, 0, 1, 24)
    full_sarima_model = SARIMAX(series,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    full_sarima_result = full_sarima_model.fit()
    today_forecast = full_sarima_result.forecast(steps=1)
    predicted_today_close = today_forecast.values[0]
    actual_today_close = today_data['Close'].item()
    sarima_accuracy = (1 - abs(actual_today_close - predicted_today_close) / actual_today_close) * 100
    return predicted_today_close, actual_today_close, sarima_accuracy
def generate_signals(y_true, y_pred, threshold=0.01):
    signals = []
    for actual, pred in zip(y_true, y_pred):
        change = (pred - actual) / actual
        if change > threshold:
            signals.append('Buy')
        elif change < -threshold:
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals


def backtest_strategy_from_model(model, X_test, y_test, threshold=0.01, initial_cash=1000):
    y_true = y_test.values.flatten()
    y_pred = model.predict(X_test)

    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()

    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    signals = generate_signals(y_true, y_pred)

    df_signals = pd.DataFrame({
        'Actual_Close': y_true,
        'Predicted_Close': y_pred,
        'Signal': signals
    }, index=y_test.index[:min_length])

    cash = initial_cash
    shares = 0
    portfolio_values = []

    for _, row in df_signals.iterrows():
        signal = row['Signal']
        price = row['Actual_Close']

        if signal == 'Buy' and cash > 0:
            shares = cash / price
            cash = 0
        elif signal == 'Sell' and shares > 0:
            cash = shares * price
            shares = 0

        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)

    df_signals['Portfolio_Value'] = portfolio_values
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    return df_signals, final_value, total_return
