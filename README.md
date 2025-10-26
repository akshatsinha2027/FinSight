# FinSight – Automated Stock Forecasting Platform
## Overview
FinSight is an automated stock price forecasting system that predicts future stock trends using hybrid deep learning and time-series models.
It integrates GRU, LSTM, Prophet, and ARIMA, with a fully automated pipeline powered by Apache Airflow and a Streamlit dashboard for visualization.
## Key Features
 Deep Learning Forecasting: GRU achieves MAPE of 2.8% — outperforming Prophet and ARIMA.
 Feature Engineering: Incorporates RSI, MACD, and moving averages (10/20/50/100/200 days) for trend analysis.
 Automation: Airflow DAG automates data fetching → preprocessing → model training → prediction.
 Interactive Dashboard: Streamlit app visualizes real-time forecasts, trends, and evaluation metrics.
 Hybrid Modeling: Combines Prophet + GRU (MAPE = 9.5%) for benchmark comparison.
## Tech Stack
Languages: Python
Libraries: TensorFlow, Keras, Prophet, ARIMA, Pandas, NumPy
Automation: Apache Airflow
Visualization: Matplotlib, Streamlit
## Model Comparison
Model	Type	Metric	MAPE
ARIMA	Statistical	15.9%	
Prophet	Additive Time Series	14.7%	
LSTM	Deep Learning	4.4%	
GRU	Deep Learning (Final)	2.8%	
Prophet + GRU	Hybrid	9.5%	
## Business Impact
FinSight automates the end-to-end forecasting workflow, allowing analysts to monitor trends, evaluate model performance, and make informed investment decisions — without manual intervention.
## Future Enhancements
Integrate live market APIs for real-time updates.
Add Explainability (SHAP/LIME) for model transparency.
Extend to multi-stock portfolio predictions.
## Demo & Documentation
GRU Model Notebook
Streamlit App Link
Airflow DAG Code
## Author
Akshat Sinha
Integrated M.Tech, IIT (ISM) Dhanbad
Aspiring ML Engineer / Data Scientist