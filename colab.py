# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense

# import pandas as pd

# # Load the dataset from a CSV file
# # file_path = "F:\\Windsor Study\\SEM 1\\A.I\\Project\\Code\\data.csv"  # Replace 'your_file_path.csv' with the path to your CSV file
# # df = pd.read_csv(file_path)

# # # Convert 'Year' column to datetime format
# # df["Year"] = pd.to_datetime(df["Year"], format="%Y")
# # df.set_index("Year", inplace=True)

# # Display the first few rows of the dataset
# # print(df.head())

# # Visualize the data
# # plt.figure(figsize=(10, 6))
# # plt.plot(df.index, df['Number of people'], marker='o', linestyle='-')
# # plt.title('Number of People Over Time')
# # plt.xlabel('Year')
# # plt.ylabel('Number of People')
# # plt.grid(True)
# # plt.show()


# # Load the dataset
# # data = {
# #     "Year": [
# #         2003,
# #         2005,
# #         2007,
# #         2008,
# #         2009,
# #         2010,
# #         2011,
# #         2012,
# #         2013,
# #         2014,
# #         2015,
# #         2016,
# #         2017,
# #         2018,
# #         2019,
# #         2020,
# #         2021,
# #         2022,
# #     ],
# #     "Number of people": [
# #         6085126,
# #         5874689,
# #         6112442,
# #         6009311,
# #         5730321,
# #         5967259,
# #         5764843,
# #         5933095,
# #         5722635,
# #         5410937,
# #         5344100,
# #         5160800,
# #         5006100,
# #         4926800,
# #         4684400,
# #         4159800,
# #         3830200,
# #         3804200,
# #     ],
# # }
# # df = pd.DataFrame(data)

# df = pd.read_csv('F:\\Windsor Study\\SEM 1\\A.I\\Project\\Code\\data.csv', delimiter='\t')

# # Automatically detect column names
# column_names = df.columns[0].split(',')

# # Split the column into 'Year' and 'Number of people'
# df[column_names] = df[df.columns[0]].str.split(',', expand=True)

# # Drop the original combined column
# df.drop(columns=[df.columns[0]], inplace=True)

# # Display the DataFrame
# print(df)

# # Feature Scaling
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(df["Number of people"].values.reshape(-1, 1))


# # Prepare the data for LSTM
# def create_dataset(data, time_steps):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i : (i + time_steps), 0])
#         y.append(data[i + time_steps, 0])
#     return np.array(X), np.array(y)


# time_steps = 3
# X_lstm, y_lstm = create_dataset(scaled_data, time_steps)
# X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# # Split the data into train and test sets
# X_train_lstm, X_test_lstm = X_lstm[:-2], X_lstm[-2:]
# y_train_lstm, y_test_lstm = y_lstm[:-2], y_lstm[-2:]

# # Build LSTM model
# model_lstm = Sequential()
# model_lstm.add(
#     LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1))
# )
# model_lstm.add(LSTM(units=50))
# model_lstm.add(Dense(units=1))
# model_lstm.compile(optimizer="adam", loss="mean_squared_error")

# # Train LSTM model
# model_lstm.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=1, verbose=0)

# # Predictions using LSTM for 2021 and 2022
# y_pred_lstm = model_lstm.predict(X_test_lstm)
# y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

# # Print the generated values
# print("Predicted values for 2021 and 2022:")
# for i, year in enumerate([2021, 2022]):
#     print(f"{year}: {int(y_pred_lstm[i])}")

# # Visualize the predictions
# plt.figure(figsize=(10, 6))
# plt.plot(df["Year"][:-2], df["Number of people"][:-2], label="Train")
# plt.plot(df["Year"][-2:], df["Number of people"][-2:], label="Test", marker="o")

# plt.plot(
#     [2021, 2022],
#     y_pred_lstm,
#     label="LSTM Predictions",
#     linestyle="--",
#     color="blue",
#     marker="x",
# )
# plt.title("LSTM Predictions vs Actual")
# plt.xlabel("Year")
# plt.ylabel("Number of People")
# plt.legend(loc="best")
# plt.grid(True)
# plt.show()


# ------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the dataset
df = pd.read_csv('F:\\Windsor Study\\SEM 1\\A.I\\Project\\Code\\data.csv', delimiter='\t')

# Automatically detect column names
column_names = df.columns[0].split(',')

# Split the column into 'Year' and 'Number of people'
df[column_names] = df[df.columns[0]].str.split(',', expand=True)

# Drop the original combined column
df.drop(columns=[df.columns[0]], inplace=True)

# Display the DataFrame
print(df)

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Number of people"].values.reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : (i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 3
X_lstm, y_lstm = create_dataset(scaled_data, time_steps)
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# Split the data into train and test sets
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
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

# Print the generated values
print("Predicted values for 2021 and 2022:")
for i, year in enumerate([2021, 2022]):
    print(f"{year}: {int(y_pred_lstm[i])}")

# Visualize the predictions
plt.figure(figsize=(10, 6))
plt.plot(df["Year"][:-2], df["Number of people"][:-2], label="Train")
plt.plot(df["Year"][-2:], df["Number of people"][-2:], label="Test", marker="o")

plt.plot(
    [2021, 2022],
    y_pred_lstm,
    label="LSTM Predictions",
    linestyle="--",
    color="blue",
    marker="x",
)
plt.title("LSTM Predictions vs Actual")
plt.xlabel("Year")
plt.ylabel("Number of People")
plt.legend(loc="best")
plt.grid(True)
plt.show()
