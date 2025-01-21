import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential

# Load the dataset
df = pd.read_csv('/Users/nishant/Downloads/Code/data.csv', delimiter='\t')
column_names = df.columns[0].split(',')
df[column_names] = df[df.columns[0]].str.split(',', expand=True)
df.drop(columns=[df.columns[0]], inplace=True)

# Feature Scaling for LSTM
scaler_lstm = MinMaxScaler(feature_range=(0, 1))
scaled_data_lstm = scaler_lstm.fit_transform(df["Number of people"].values.reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : (i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps_lstm = 3
X_lstm, y_lstm = create_dataset(scaled_data_lstm, time_steps_lstm)
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# Split the data into train and test sets for LSTM
X_train_lstm, X_test_lstm = X_lstm[:-2], X_lstm[-2:]
y_train_lstm, y_test_lstm = y_lstm[:-2], y_lstm[-2:]

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer="adam", loss="mean_squared_error")

# Train LSTM model
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=1, verbose=0)

# Predictions using LSTM for 2021 and 2022
y_pred_lstm = model_lstm.predict(X_test_lstm)
y_pred_lstm = scaler_lstm.inverse_transform(y_pred_lstm)

# Print the predicted values for LSTM
st.write("LSTM Predicted values for 2021 and 2022:")
for year, pred_value in zip([2021, 2022], y_pred_lstm):
    st.write(f"{year}: {int(pred_value)}")

# Feature Scaling for XGBoost
scaler_xgboost = MinMaxScaler(feature_range=(0, 1))
scaled_data_xgboost = scaler_xgboost.fit_transform(df['Number of people'].values.reshape(-1, 1))

# Prepare the data for XGBoost
time_steps_xgboost = 3
X_xgboost, y_xgboost = create_dataset(scaled_data_xgboost, time_steps_xgboost)

# Split the data into train and test sets for XGBoost
X_train_xgboost, X_test_xgboost = X_xgboost[:-2], X_xgboost[-2:]
y_train_xgboost, y_test_xgboost = y_xgboost[:-2], y_xgboost[-2:]

# Build XGBoost model
model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=1000)

# Train XGBoost model
model_xgboost.fit(X_train_xgboost, y_train_xgboost)

# Predictions using XGBoost for 2021 and 2022
y_pred_xgboost = model_xgboost.predict(X_test_xgboost)
y_pred_xgboost = scaler_xgboost.inverse_transform(y_pred_xgboost.reshape(-1, 1))

# Print the predicted values for XGBoost
st.write("XGBoost Predicted values for 2021 and 2022:")
for i, year in enumerate([2021, 2022]):
    st.write(f"{year}: {int(y_pred_xgboost[i][0])}")
    
# Convert 'Number of people' column to numeric
df['Number of people'] = pd.to_numeric(df['Number of people'], errors='coerce')

# Drop rows with NaN values, if any
df.dropna(subset=['Number of people'], inplace=True)

# Train ARIMA model
model_arima = ARIMA(df['Number of people'], order=(5,1,0))
model_arima_fit = model_arima.fit()


# Train ARIMA model
model_arima = ARIMA(df['Number of people'], order=(5,1,0))
model_arima_fit = model_arima.fit()

# Predictions using ARIMA for 2021 and 2022
y_pred_arima = model_arima_fit.forecast(steps=2)

# Print the predicted values for ARIMA
st.write("ARIMA Predicted values for 2021 and 2022:")
for year, pred_value in zip([2021, 2022], y_pred_arima):
    st.write(f"{year}: {int(pred_value)}")

# Fit the ETS model
model_ets = ExponentialSmoothing(df['Number of people'], trend='add', seasonal='add', seasonal_periods=4)
model_ets_fit = model_ets.fit()

# Predictions for the years 2021 and 2022 using ETS
predictions_2021_2022 = model_ets_fit.forecast(2)
st.write("ETS Predicted values for 2021 and 2022:")
for year, pred_value in zip([2021, 2022], predictions_2021_2022):
    st.write(f"{year}: {int(pred_value)}")


# Visualize the main dataset 
fig_pred_main = go.Figure()

# Add main dataset
fig_pred_main.add_trace(go.Scatter(
    x=df['Year'][:-2],
    y=df['Number of people'][:-2],
    mode='lines',
    name='Main Dataset',
    line=dict(color='blue')
))

st.plotly_chart(fig_pred_main, use_container_width=True)


# Visualizing the combined main data and the predictions
fig_pred_combined = go.Figure()

# Add main dataset
fig_pred_combined.add_trace(go.Scatter(
    x=df['Year'][:-2],
    y=df['Number of people'][:-2],
    mode='lines',
    name='Main Dataset',
    line=dict(color='blue')
))

# Add LSTM predictions
fig_pred_combined.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_pred_lstm.flatten(),
    mode='markers+lines',
    name='LSTM Predictions',
    marker=dict(color='green', size=12, symbol='x')
))

# Add XGBoost predictions
fig_pred_combined.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_pred_xgboost.flatten(),
    mode='markers+lines',
    name='XGBoost Predictions',
    marker=dict(color='red', size=12, symbol='x')
))

# Add ARIMA predictions
fig_pred_combined.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_pred_arima.values.flatten(),  # Convert Series to NumPy array and then flatten
    mode='markers+lines',
    name='ARIMA Predictions',
    marker=dict(color='purple', size=12, symbol='x')
))


# Add ETS predictions
fig_pred_combined.add_trace(go.Scatter(
    x=[2021, 2022],
    y=predictions_2021_2022,
    mode='markers+lines',
    name='ETS Predictions',
    marker=dict(color='orange', size=12, symbol='x')
))

fig_pred_combined.update_layout(
    title='Predictions from Different Models vs Actual',
    xaxis_title='Year',
    yaxis_title='Number of People',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Display the combined plot using Streamlit
st.plotly_chart(fig_pred_combined, use_container_width=True)

import streamlit as st
import plotly.graph_objects as go

# Scatter plot showing the actual values versus the predicted values for each model for the years 2021 and 2022
fig_scatter = go.Figure()

# LSTM scatter plot
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_test_lstm,
    mode='markers',
    name='Actual LSTM',
    marker=dict(color='blue', size=10)
))
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_pred_lstm.flatten(),
    mode='markers',
    name='Predicted LSTM',
    marker=dict(color='green', size=10)
))

# XGBoost scatter plot
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_test_xgboost,
    mode='markers',
    name='Actual XGBoost',
    marker=dict(color='red', size=10)
))
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_pred_xgboost.flatten(),
    mode='markers',
    name='Predicted XGBoost',
    marker=dict(color='orange', size=10)
))

# ARIMA scatter plot
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=df['Number of people'][-2:],
    mode='markers',
    name='Actual ARIMA',
    marker=dict(color='purple', size=10)
))
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_pred_arima,
    mode='markers',
    name='Predicted ARIMA',
    marker=dict(color='yellow', size=10)
))

# ETS scatter plot
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=df['Number of people'][-2:],
    mode='markers',
    name='Actual ETS',
    marker=dict(color='cyan', size=10)
))
fig_scatter.add_trace(go.Scatter(
    x=[2021, 2022],
    y=predictions_2021_2022,
    mode='markers',
    name='Predicted ETS',
    marker=dict(color='pink', size=10)
))

fig_scatter.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Year',
    yaxis_title='Number of People'
)

st.plotly_chart(fig_scatter, use_container_width=True)


# Error plot showing the difference between the actual values and the predicted values for each model for the years 2021 and 2022
fig_error = go.Figure()

# LSTM error plot
fig_error.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_test_lstm - y_pred_lstm.flatten(),
    mode='lines+markers',
    name='LSTM',
    line=dict(color='blue', width=2),
    marker=dict(color='blue', size=10)
))

# XGBoost error plot
fig_error.add_trace(go.Scatter(
    x=[2021, 2022],
    y=y_test_xgboost - y_pred_xgboost.flatten(),
    mode='lines+markers',
    name='XGBoost',
    line=dict(color='red', width=2),
    marker=dict(color='red', size=10)
))

# ARIMA error plot
fig_error.add_trace(go.Scatter(
    x=[2021, 2022],
    y=df['Number of people'][-2:] - y_pred_arima,
    mode='lines+markers',
    name='ARIMA',
    line=dict(color='purple', width=2),
    marker=dict(color='purple', size=10)
))

# ETS error plot
fig_error.add_trace(go.Scatter(
    x=[2021, 2022],
    y=df['Number of people'][-2:] - predictions_2021_2022,
    mode='lines+markers',
    name='ETS',
    line=dict(color='green', width=2),
    marker=dict(color='green', size=10)
))

fig_error.update_layout(
    title='Prediction Errors',
    xaxis_title='Year',
    yaxis_title='Prediction Error'
)

st.plotly_chart(fig_error, use_container_width=True)


