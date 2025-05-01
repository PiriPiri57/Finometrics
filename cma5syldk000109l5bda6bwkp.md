---
title: "Finometrics"
datePublished: Thu May 01 2025 20:13:27 GMT+0000 (Coordinated Universal Time)
cuid: cma5syldk000109l5bda6bwkp
slug: finometrics
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/fiXLQXAhCfk/upload/67681c6ad27d7a55343dddd3ae854318.jpeg
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1746130352857/547d0940-ad2c-4ce3-b801-1a6dc9862004.jpeg

---

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746130240485/ed06b589-8d62-4376-959d-6f3ff1b35f45.png align="center")

[GitHub Repo](https://github.com/riddhika05/Finometrics)

[Colab](https://colab.research.google.com/drive/10jnqBRc4eKo5CaUSS6JyqDQKvuRQQDaM?usp=sharing)

[Streamlit](https://finometrics-h8kanm66teun6dubtnuzsv.streamlit.app/)

---

# 📈 Building a Stock Market Prediction & Strategy Simulator (with Streamlit Web App!)

---

## 🚀 Introduction

Ever wondered how stock prediction models work? Or how machine learning and time-series forecasting can help simulate trading strategies?

In this blog, I'll walk you through an **end-to-end stock analysis pipeline** from fetching stock data using the `yfinance` API, building regression and classification models, forecasting with SARIMA, to backtesting trading strategies.  
We have chosen Reliance as default stock as it is a stable stock and gives good output, you can chose the ticker to any valid stock.

✨ Oh, and we wrapped it all into a **Streamlit web app** so you can interact with it live!

---

## 📥 Step 1: Fetching Real-Time Stock Data with `yfinance`

We used the popular `yfinance` library to fetch daily stock prices. But markets don’t trade on weekends or holidays, so we used `pandas_market_calendars` to get the **last valid trading day** using the NSE calendar.

```python
df = yf.download(ticker, end=(last_working_day + timedelta(days=1)).strftime('%Y-%m-%d'))
```

### ⚠️ Beware of `yfinance` Pitfalls

Sometimes, too many API calls result in:

* Rate limit errors
    
* Empty dataframes
    

**Solution:**  
✅ Fallback to a local CSV if the API fails

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746130775935/e7e754ad-75ed-40a2-9739-0bf9e474db92.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746130846082/4f485198-0fd9-497c-b770-8f01329be116.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746130947809/95642ec3-22ca-4088-81ac-c6173653b773.png align="center")

---

## 🧹 Step 2: Data Cleaning + Feature Engineering

To make the model smarter, we extracted time-based and technical indicators:

✅Check for Null Values:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131635732/e42f53bd-c744-4029-a44a-0ebc1fd1d794.png align="center")

📅 Time Features:

* Year, Month, Day of Week, Month Start/End
    

📊 Price Signals:

* `lag_1`, `lag_5`, `lag_10`
    
* `rolling_mean_5`, `rolling_mean_10`, `rolling_mean_20`
    
* `Log_Return`, `Daily_Return`
    

📦 Volume Signals:

* `Log_Volume`, `Rolling_Volume`
    

After creating the features, we dropped:

* Redundant columns (e.g., `Open`, `Low`,`Dividends`,`Stock Splits`)
    
* Categorical columns(e.g., `Date`)
    
* Highly correlated features (to prevent overfitting)
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131029781/3300b10b-a245-47a4-97ed-4681e529c5b0.png align="center")
    

---

## 📊 Step 3: Exploratory Data Analysis (EDA)

### 🔥 Correlation Heatmap

We used seaborn’s heatmap to visualize feature correlations and eliminated multicollinear variables.

```python
sns.heatmap(data=df.corr(), annot=True)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746130727602/adceca9a-1dc8-44ff-bdf6-e138d9411187.png align="center")

### 🧪 Outlier Detection

Outliers were visualized using boxplots—but **not removed**, as they often indicate real market movements!

```python
sns.boxplot(df['Close'])    #Checking for outliers, because for stock market outliers indicate trend breakers
sns.displot(df['Close'])    #Eventough we have outliers, we wont remove it as they carry valuable information about the stock
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131140970/a7b1f3ea-142b-45b3-8c92-33e2ec28551b.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131162141/0d1f555e-1919-4458-97c3-a46a06430aa8.png align="center")

## 🤖 Step 4: Predicting Stock Prices — Regression Models

We trained multiple regression models to predict **today's closing price**:

* Linear, Ridge, Lasso Regression
    
* Random Forest
    
* Gradient Boosting
    
* XGBoost
    

📏 Evaluation Metrics:

* MAE (Mean Absolute Error)
    
* RMSE (Root Mean Squared Error)
    
* R² Score
    

🎯 **Best model** was selected based on highest R².

```python
X = df.drop(['Close'], axis=1)
y = df['Close']

# Sort the data by date
df = df.sort_index()

#split data into train and test data (80 % training data & 20 % test data)  
#in stock models we dont randomly select data for training and splitting rather assign in this way
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Use some Well known Models
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

results_df = pd.DataFrame(results).T
print(results_df)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131255885/2d035402-162d-4962-9028-5855739a2756.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131314506/3ae9f71e-9d4e-4f97-a517-6bf4080861c1.png align="center")

## 🤔**Predicting Today’s Close Value:**

### 🎯We will automatically select the best-performing model and use it for regression to obtain the result.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131530799/ca08879f-c726-4a01-88de-4bde0cbc6452.png align="center")

## 🎯 Step 5: Gain or Fall? Let’s Classify!

We reframed the problem:  
👉 Will the stock **gain (1)** or **fall (0)** tomorrow?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131813189/e1ba1053-18da-4f83-b57a-8e737ef8e94a.png align="center")

Trained models:

* Logistic Regression
    
* Random Forest
    
* XGBoost
    
* KNN
    
* Decision Tree
    
* Gradient Boosting
    
    ```python
    
    models_class = [LogisticRegression(),
      XGBClassifier(),
      RandomForestClassifier(n_estimators=100, random_state=42),
      DecisionTreeClassifier(random_state=42),                      #Some Common Classifiers
      KNeighborsClassifier(n_neighbors=5),
      GradientBoostingClassifier(n_estimators=100, random_state=42)]
    
    validation_scores = {}
    
    for i in range(6):
      models_class[i].fit(X_train_cls, y_train_cls)
    
      print(f'{models_class[i].__class__.__name__} : ')
      print('Training Accuracy : ', metrics.roc_auc_score(
        y_train_cls, models_class[i].predict_proba(X_train_cls)[:,1]))
      print('Validation Accuracy : ', metrics.roc_auc_score(
        y_test_cls, models_class[i].predict_proba(X_test_cls)[:,1]))
      print()
      validation_scores[models_class[i].__class__.__name__] = metrics.roc_auc_score(
        y_test_cls, models_class[i].predict_proba(X_test_cls)[:,1])
    
    best_model_class_name = max(validation_scores, key=validation_scores.get)
    best_model_class = models_class[[model.__class__.__name__ for model in models_class].index(best_model_class_name)]
    
    print(f"\nSelected Best Model: {best_model_class_name} with validation score: {validation_scores[best_model_class_name]:.4f}")  #In Stock market even the Market Guru's dont get more than 60% accuracy, so we are almost there😅
    ```
    

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131885079/3ecff8dd-b411-4f39-9a74-b26260417d87.png align="center")

📈 ROC-AUC was the main metric to compare classifier performance.  
🎖 The best classifier achieved &gt;55% AUC—respectable in financial modeling.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746131924480/3584565c-3f1d-4d7d-8d6c-7235e1bac9e7.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746134981112/ad5c2cf7-21c3-4c17-9a66-491133ac3fce.png align="center")

---

## 🔮 Step 6: Forecasting with SARIMA

SARIMA is perfect for seasonal time series like stocks.

### 🔍What We Did:

* Analyzed ACF/PACF plots to choose `order=(1,0,1)` and `seasonal_order=(1,0,1,24)`
    
* Forecasted:
    
    * Today’s price
        
    * Next **365 business days** (1 year!)
        
        ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746135024086/6112f8a4-b137-4218-b3a2-2108befb6a22.png align="center")
        

### 📊 Outputs:

* Predicted vs Actual Price
    
* 1-Year Forward Chart
    

```python
order = (1, 0, 1)
seasonal_order = (1, 0, 1, 24) #seasonality is 24 which means 2 years

sarima_model = SARIMAX(train,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarima_result = sarima_model.fit()

n_periods = len(test)
forecast = sarima_result.forecast(steps=n_periods)

rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'SARIMA RMSE: {rmse:.2f}')

plt.figure(figsize=(14,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, color='green', label='Test')
plt.plot(test.index, forecast, color='red', label='Forecast (SARIMA)')
plt.title('SARIMA Forecast vs Actuals')
plt.legend()
plt.grid(True)
plt.show()
```

---

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746135185014/8b3e16f7-ff6a-4a8a-9660-cd3639238b70.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746135554995/0b6427de-d5f8-4f96-8aec-92711d96408f.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746135608522/a04cde14-71f3-42b8-8d63-6ca0ff85c34e.png align="center")

## 🪙 Step 7: Buy/Sell/Hold Strategy Generation

Based on price prediction:

* **Buy** if predicted price is &gt;1% higher
    
* **Sell** if predicted price is &gt;1% lower
    
* **Hold** otherwise
    

```python
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
```

We visualized:

* Signal distribution
    
* Strategy outputs over time
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746135644805/97b647f7-83fc-4a8c-8f25-d5d0b86db085.png align="center")
    

---

## 💸 Step 8: Strategy Backtesting

We simulated a trader starting with ₹1000 using our signals:

📈 Portfolio Value was updated at every time step.

### 💹Results:

* Final Portfolio Value: `₹2631.47`
    
* Total Return: `163.15%`
    

🧪 All simulated no real money was gained/harmed.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746135757376/762e0d85-64a9-4938-a7a7-41cde7c48ee9.png align="center")

---

## 🌐 Step 9: Deploying as a Streamlit Web App

We bundled the entire project into an easy-to-use **Streamlit dashboard** 🎉

### 💡 Features:

✅ Enter any stock ticker (e.g., `RELIANCE.NS`)  
✅ Predict **gain/fall**  
✅ Forecast **closing price**  
✅ Predict **with SARIMA**  
✅ Simulate **1-year price forecast**  
✅ Backtest your strategy live  
✅ All visualized with clean plots!

```python
st.title("📈 Stock Price Prediction Dashboard")
```

📷 **Preview:**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746135961085/da5074ca-72e7-4bf7-b2f9-5c81c8fb86ed.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746136093403/7c3c96bf-f0b6-4af2-8060-41a633c5a356.png align="center")

Keep changing the ticker to get information about your desired stock😎

The complete code for `app.py` and `utils.py` is available on our GitHub repository.

[app.py](https://github.com/riddhika05/Finometrics/blob/main/app.py)

[utils.py](https://github.com/riddhika05/Finometrics/blob/main/utils.py)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746136423912/fd12efa0-83e1-4ae1-b4cc-a6203f61adb4.png align="center")

---

## 🔧 Tech Stack

* Python 🐍
    
* Streamlit 🎛️
    
* yfinance 📈
    
* Scikit-learn 🤖
    
* statsmodels 🔮
    
* XGBoost 🚀
    
* Matplotlib & Seaborn 📊
    

---

## ✅ Conclusion

From data ingestion to full deployment, this project shows the **power of data science in finance**. It’s not just about predictions it’s about building systems that help make **informed decisions**.

You now have:

* A forecasting engine
    
* A trading simulator
    
* A web interface
    

All customizable and ready to expand 🚀

---

## 💬 Want to Try It?

Deploy it locally or on the cloud via:

```bash
streamlit run app.py
```

Or fork the code and host it on platforms like **Streamlit Cloud**, **Heroku**, or **Render**.

Note:  
It works perfectly well while hosting it locally, but while deploying it, yfinance api runs into a problem namely DDOS, hence we have used a fall back csv file, it has dataset of Reliance

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1746136513530/ea6f2903-bb71-41be-8e14-0848da832706.png align="center")

---

## 🧠 Future Scope

* 📊 Add support for cryptocurrencies
    
* ⏱️ Enable real-time predictions using live market data
    
* 🤖 Integrate with trading APIs for automated execution
    
* 🗞️ Incorporate sentiment analysis from news and social media
    
* 🔁 Implement LSTM/GRU models for deep learning-based forecasting
    

---