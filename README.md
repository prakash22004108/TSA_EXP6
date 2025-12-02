## Devloped by:PRAKASH R
## Register Number: 212222240074
## Date: 17-11-2025

# Ex.No: 6                   HOLT WINTERS METHOD

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model  predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:

```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset and perform data exploration
# Make sure 'heart_rate.csv' is in the same directory as your script
data = pd.read_csv('/content/heart_rate.csv')

# --- Select the column to analyze ---
# You can change 'T1' to 'T2', 'T3', or 'T4'
time_series_column = 'T1'
time_series_data = data[time_series_column].dropna()

print("Original Data Head:")
print(data.head())


# Plot the data
plt.figure(figsize=(12, 6))
time_series_data.plot()
plt.title(f'Heart Rate Time Series ({time_series_column})')
plt.xlabel('Time Step')
plt.ylabel('Heart Rate')
plt.savefig('heart_rate_timeseries.png')


# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(time_series_data.values.reshape(-1, 1)).flatten(), index=time_series_data.index)

# Check for seasonality
# We assume a seasonal period of 12. You may need to adjust this.
decomposition = seasonal_decompose(time_series_data, model="additive", period=12)
decomposition_plot = decomposition.plot()
plt.savefig('seasonal_decomposition.png')


# Split data into training and testing sets
scaled_data = scaled_data + 1 # Add 1 to handle multiplicative seasonality
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Create and train the Holt-Winters model
model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Evaluate the model
test_predictions = model.forecast(steps=len(test_data))

plt.figure(figsize=(12, 6))
train_data.plot(legend=True, label='Train Data')
test_data.plot(legend=True, label='Test Data')
test_predictions.plot(legend=True, label='Test Predictions')
plt.title('Model Evaluation')
plt.savefig('model_evaluation.png')


# Calculate and print the Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print(f"\nRoot Mean Squared Error: {rmse}")

# Create the final model and forecast future data
final_model = ExponentialSmoothing(time_series_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
future_forecast = final_model.forecast(steps=int(len(time_series_data)/4))

# Plot the final predictions
plt.figure(figsize=(12, 6))
time_series_data.plot(legend=True, label='Original Data')
future_forecast.plot(legend=True, label='Future Forecast')
plt.title('Future Heart Rate Forecast')
plt.xlabel('Time Step')
plt.ylabel('Heart Rate')
plt.savefig('final_prediction.png')

plt.show()
```
### OUTPUT:
 <img width="389" height="152" alt="image" src="https://github.com/user-attachments/assets/66c906cc-9790-499e-bab9-a0ecd7917b92" />
<img width="959" height="511" alt="image" src="https://github.com/user-attachments/assets/b247486b-3f52-4afe-bc49-bba6ca4613bc" />
<img width="600" height="434" alt="image" src="https://github.com/user-attachments/assets/0748b41a-1e59-4be2-91c5-bd2d0531da87" />
<img width="947" height="487" alt="image" src="https://github.com/user-attachments/assets/a534db56-035d-4ed0-a6bd-f475daeaafd4" />
<img width="976" height="508" alt="image" src="https://github.com/user-attachments/assets/7f31eed2-da45-4278-b2a7-8358147f5841" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
