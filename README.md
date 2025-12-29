# 📊 Crypto Trend Prediction & Financial Data Analysis App

## 🔍 Overview

A Streamlit-based application for analyzing cryptocurrency price data and forecasting short-term trend direction using **Machine Learning (Random Forest)** combined with **trading strategy backtesting** and **Walk-Forward Optimization**.

Current sample dataset focus: **ETH/USDT (30-minute timeframe)**.

## ⚙️ Key Features

* **Data Loading & Quality Checks**

  * Supports CSV / Excel upload
  * Duplicate & missing timestamp detection
  * Date range & descriptive statistics

* **Exploratory Data Analysis (EDA)**

  * Return distribution (Histogram, Boxplot)
  * Local trend via rolling mean of returns
  * Volatility measurement via rolling std

* **Feature Engineering**

  * Price action features: upper/lower shadow, gap ratios
  * Technical indicators: RSI14, MACD histogram, ATR normalized
  * Volume shock and spike-weighted ATR
  * All features scaled with **RobustScaler**

* **Trend Forecasting**

  * Labels next-period movement as **Up (1)** or **Down (-1)**
  * Model trained using **RandomForestClassifier**
  * Probability delta smoothing (EMA-based) to generate trading signals

* **Trading Strategy & Backtesting**

  * ML signals converted into **long/exit positions**
  * Evaluated using:

    * Return %, Sharpe Ratio, Max Drawdown, Profit Factor, Win Rate, Total Trades

* **Walk-Forward Optimization & Statistical Validation**

  * Sequential training/testing segments
  * Performance comparison vs default parameters
  * Significance testing using:

    * **T-test (mean > 0)**
    * **Wilcoxon (optimized vs default)**

## 🛠 Tech Stack

| Category          | Tools                                          |
| ----------------- | ---------------------------------------------- |
| Frontend          | Streamlit                                      |
| Data Processing   | Pandas, NumPy                                  |
| Visualization     | Matplotlib                                     |
| ML Model          | Scikit-Learn (Random Forest)                   |
| Backtesting       | backtesting.py                                 |
| Optimization      | Hyperopt (TPE)                                 |
| Statistical Tests | SciPy (T-test, Wilcoxon, Wilcoxon signed-rank) |

## 📥 Data Format Requirements

Your dataset should include the following columns:

```
timestamp, open, high, low, close, volume
```

(Timestamp must be convertible to `datetime`)


## 📈 Use Case

This project is suitable for:

* ML-based trend direction research
* Crypto trading strategy prototyping
* Probability-driven entry/exit signal modeling
* Walk-forward performance evaluation without look-ahead bias

## ⚠️ Notes

* Uses **train/test split sequentially** to avoid look-ahead bias
* Signals are generated using **in-sample probability delta**, then smoothed to stabilize trade decisions

## 👤 Author

**Nguyen Tuan Phat**
Financial Technology • Banking & Crypto ML Research
