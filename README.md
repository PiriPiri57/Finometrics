---

# Stock Market Analysis, Prediction, and Strategy Simulation

Welcome to the Stock Market Prediction project.  
This project uses real-time stock data to predict stock prices, classify market movements (gain/fall), and simulate trading strategies using machine learning and time series forecasting models,namely SARIMA.

---

## About This Project

This project focuses on *analyzing and predicting the stock price of Reliance Industries Ltd. (Ticker: RELIANCE.NS)* listed on the National Stock Exchange (NSE).

It covers:
- Real-time stock data fetching using yfinance
- Handling NSE trading calendars to account for weekends and holidays
- Feature Engineering including:
  - Lag Features
  - Rolling Means
  - Logarithmic Returns
  - Volume Shifts
- Predicting Stock Closing Price using various regression models
- Classifying Next-Day Gain or Fall using classification models
- Forecasting Future Stock Prices using SARIMA model
- Simulating a Trading Strategy based on Buy/Sell/Hold signals
- Backtesting Portfolio Growth and calculating returns

---

## Key Highlights

- Real-Time Stock Data fetching from Yahoo Finance (yfinance)
- Feature Engineering:
  - Lag features, rolling means, logarithmic returns, volume trends
- Price Prediction:
  - Regression models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
- Market Movement Classification:
  - Classifiers: Logistic Regression, XGBoost, Random Forest, KNN, Gradient Boosting
- Advanced Time-Series Forecasting:
  - SARIMA model for accurate multi-day forecasting
- Trading Strategy Backtesting:
  - Portfolio simulation using predicted Buy/Sell/Hold signals

---

## Tech Stack

- Language: Python
- Libraries:
  - Data Handling: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Machine Learning: scikit-learn, xgboost
  - Time Series Modeling: statsmodels
  - Data Sources: yfinance, pandas_market_calendars
- Development Environment: Google Colab / Jupyter Notebook

---

## Workflow Overview

1. Data Acquisition:
   - Download historical and latest stock data
2. Data Cleaning and Feature Engineering:
   - Generate lagged features, rolling means, returns
   - Handle missing data and anomalies
3. Model Building:
   - Train multiple ML models and select the best performers
   - Evaluate using MAE, RMSE, R² Score, and AUC
4. Forecasting Future Stock Prices:
   - Apply SARIMA for 365-day close price forecasting
5. Trading Strategy Development:
   - Generate trading signals and simulate portfolio growth

---

## Results

| Model               | MAE   | RMSE  | R² Score |
|---------------------|-------|-------|----------|
| Ridge Regression     | Best  | Low   | High     |
| XGBoost Classifier   | -     | -     | 0.97+    |
| SARIMA Forecasting   | -     | ~Low Error | Accurate Daily + Long-Term |

- Price Prediction Accuracy: ~97–98%
- Classification ROC-AUC: ~95–97%
- Backtested Portfolio Return: ~24% profit

---

## Getting Started

### Installation

bash
pip install yfinance pandas matplotlib seaborn scikit-learn xgboost statsmodels pandas_market_calendars


### Running the Project

- Open the notebook or script
- Modify the ticker symbol if needed
- Run all cells/scripts sequentially

---

## Visualizations

- Correlation Heatmaps
- Close Price vs Time Graphs
- ROC Curves for Classifiers
- Portfolio Value Growth Curve
- 365-Day Future Forecast Charts
- Signal Distribution (Buy/Sell/Hold)

---

## Future Enhancements

- Integrate live stock predictions in real-time
- Expand to portfolio optimization across multiple stocks
- Implement Deep Learning models (LSTM, GRU)
- Deploy as a Flask/Django-based stock prediction web app

---

## Acknowledgments

- Yahoo Finance for providing open stock data  
- scikit-learn and XGBoost for modeling  
- Statsmodels for SARIMA time series models  

---
## Team Members

- Riddhika Arora
- Priyanshu Kumar
- Ojas Samar

## Open in Google Colab

[Open in Colab](https://colab.research.google.com/github/riddhika05/Finometrics/blob/main/Stock_Analysis.ipynb)



---

 **Live Demo**:  
 [Click here to open the app](https://finometrics-h8kanm66teun6dubtnuzsv.streamlit.app/)