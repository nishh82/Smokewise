import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
df = pd.read_csv('F:\\Windsor Study\\SEM 1\\A.I\\Project\\Code\\data.csv', delimiter='\t')
column_names = df.columns[0].split(',')
df[column_names] = df[df.columns[0]].str.split(',', expand=True)
df.drop(columns=[df.columns[0]], inplace=True)

app = dash.Dash(__name__)

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

# Convert 'Number of people' column to numeric
df['Number of people'] = pd.to_numeric(df['Number of people'], errors='coerce')

# Drop rows with NaN values, if any
df.dropna(subset=['Number of people'], inplace=True)

# Train ARIMA model
model_arima = ARIMA(df['Number of people'], order=(5,1,0))
model_arima_fit = model_arima.fit()

# Predictions using ARIMA for 2021 and 2022
y_pred_arima = model_arima_fit.forecast(steps=2)

# Fit the ETS model
model_ets = ExponentialSmoothing(df['Number of people'], trend='add', seasonal='add', seasonal_periods=4)
model_ets_fit = model_ets.fit()

# Predictions for the years 2021 and 2022 using ETS
predictions_2021_2022 = model_ets_fit.forecast(2)

app.layout = html.Div([
    html.H1('Predictions from Different Models vs Actual'),
    
    dcc.Graph(
        id='combined-predictions',
        figure={
            'data': [
                {'x': df['Year'][:-2], 'y': df['Number of people'][:-2], 'type': 'lines', 'name': 'Main Dataset', 'line': {'color': 'blue'}},
                {'x': [2021, 2022], 'y': y_pred_lstm.flatten(), 'mode': 'markers+lines', 'name': 'LSTM Predictions', 'marker': {'color': 'green', 'size': 12, 'symbol': 'x'}},
                {'x': [2021, 2022], 'y': y_pred_xgboost.flatten(), 'mode': 'markers+lines', 'name': 'XGBoost Predictions', 'marker': {'color': 'red', 'size': 12, 'symbol': 'x'}},
                {'x': [2021, 2022], 'y': y_pred_arima.values.flatten(), 'mode': 'markers+lines', 'name': 'ARIMA Predictions', 'marker': {'color': 'purple', 'size': 12, 'symbol': 'x'}},
                {'x': [2021, 2022], 'y': predictions_2021_2022, 'mode': 'markers+lines', 'name': 'ETS Predictions', 'marker': {'color': 'orange', 'size': 12, 'symbol': 'x'}}
            ],
            'layout': {
                'title': 'Predictions from Different Models vs Actual',
                'xaxis': {'title': 'Year'},
                'yaxis': {'title': 'Number of People'},
                'legend': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'left', 'x': 0.01}
            }
        }
    ),
    
    html.H1('Actual vs Predicted Values'),
    
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                {'x': [2021, 2022], 'y': y_test_lstm, 'mode': 'markers', 'name': 'Actual LSTM', 'marker': {'color': 'blue', 'size': 10}},
                {'x': [2021, 2022], 'y': y_pred_lstm.flatten(), 'mode': 'markers', 'name': 'Predicted LSTM', 'marker': {'color': 'green', 'size': 10}},
                {'x': [2021, 2022], 'y': y_test_xgboost, 'mode': 'markers', 'name': 'Actual XGBoost', 'marker': {'color': 'red', 'size': 10}},
                {'x': [2021, 2022], 'y': y_pred_xgboost.flatten(), 'mode': 'markers', 'name': 'Predicted XGBoost', 'marker': {'color': 'orange', 'size': 10}},
                {'x': [2021, 2022], 'y': df['Number of people'][-2:], 'mode': 'markers', 'name': 'Actual ARIMA', 'marker': {'color': 'purple', 'size': 10}},
                {'x': [2021, 2022], 'y': y_pred_arima, 'mode': 'markers', 'name': 'Predicted ARIMA', 'marker': {'color': 'yellow', 'size': 10}},
                {'x': [2021, 2022], 'y': df['Number of people'][-2:], 'mode': 'markers', 'name': 'Actual ETS', 'marker': {'color': 'cyan', 'size': 10}},
                {'x': [2021, 2022], 'y': predictions_2021_2022, 'mode': 'markers', 'name': 'Predicted ETS', 'marker': {'color': 'pink', 'size': 10}}
            ],
            'layout': {
                'title': 'Actual vs Predicted Values',
                'xaxis': {'title': 'Year'},
                'yaxis': {'title': 'Number of People'}
            }
        }
    ),
    
    html.H1('Prediction Errors'),
    
    dcc.Graph(
        id='error-plot',
        figure={
            'data': [
                {'x': [2021, 2022], 'y': y_test_lstm - y_pred_lstm.flatten(), 'mode': 'lines+markers', 'name': 'LSTM', 'line': {'color': 'blue', 'width': 2}, 'marker': {'color': 'blue', 'size': 10}},
                {'x': [2021, 2022], 'y': y_test_xgboost - y_pred_xgboost.flatten(), 'mode': 'lines+markers', 'name': 'XGBoost', 'line': {'color': 'red', 'width': 2}, 'marker': {'color': 'red', 'size': 10}},
                {'x': [2021, 2022], 'y': df['Number of people'][-2:] - y_pred_arima, 'mode': 'lines+markers', 'name': 'ARIMA', 'line': {'color': 'purple', 'width': 2}, 'marker': {'color': 'purple', 'size': 10}},
                {'x': [2021, 2022], 'y': df['Number of people'][-2:] - predictions_2021_2022, 'mode': 'lines+markers', 'name': 'ETS', 'line': {'color': 'green', 'width': 2}, 'marker': {'color': 'green', 'size': 10}}
            ],
            'layout': {
                'title': 'Prediction Errors',
                'xaxis': {'title': 'Year'},
                'yaxis': {'title': 'Prediction Error'}
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
