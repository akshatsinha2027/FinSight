# FinSight – Stock Price Forecasting App
Deployed App: https://finsightdemo.streamlit.app
Tech Stack: Python · TensorFlow · Prophet · ARIMA · GRU · Streamlit
## Overview
FinSight is a deep learning–powered stock forecasting app that predicts the next-k-day closing prices of any listed stock using GRU and hybrid Prophet models.
It combines classical time-series forecasting with modern deep learning to deliver accurate, interactive predictions for retail investors and analysts.
## Key Features
Data Ingestion: Fetches real-time data from Yahoo Finance API.
Feature Engineering: Generates MACD, RSI, SMA/EMA, and lag features for trend detection.
Modeling:
ARIMA & Prophet (classical models)
LSTM & GRU (deep learning models)
Prophet + GRU hybrid ensemble
Evaluation: Mean Absolute Percentage Error (MAPE) and visual comparison plots.
Deployment: Interactive Streamlit app for real-time forecasting & visualization.
## Architecture
Data Fetching → Feature Engineering → Model Training (ARIMA, Prophet, GRU)
                     ↓
           Hybrid Forecast Generation
                     ↓
       Visualization via Streamlit Dashboard
## Tech Stack
1.Data Source --> Yahoo Finance (via yfinance)
2.Modeling --> ARIMA, Prophet, LSTM, GRU
3.Frameworks --> TensorFlow, Scikit-learn
4.Visualization --> Matplotlib, Streamlit
5.Languages --> Python
6.Deployment --> Streamlit Community Cloud
## Results
Model	Metric	Performance
GRU	MAPE	2.8%
Hybrid Prophet + GRU	MAPE	9.5%
ARIMA	MAPE	15.2%
Prophet	MAPE	12.1%
--> GRU outperformed all others, showing strong short-term prediction accuracy.
## App Interface
User Input: Enter any stock ticker symbol (e.g., AAPL, MSFT, TSLA).
Outputs:
Predicted vs. Actual plots
MAPE scores
Next-k-day forecast table
## Learning Takeaways
Implemented end-to-end time-series forecasting pipeline from raw data to deployment.
Gained experience in deep learning sequence models (GRU, LSTM).
Learned app deployment and visualization using Streamlit.
## Future Enhancements
Add automatic daily retraining pipeline.
Integrate Docker and Airflow for scheduling.
Connect to a cloud database for persistent forecasts.
Explore transformer-based forecasting (Informer / Temporal Fusion Transformer).
## Author
Akshat Sinha
Data Science & AI Enthusiast | IIT (ISM) Dhanbad