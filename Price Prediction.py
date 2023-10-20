#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Use pip install yfinance pandas to get yfinance

import os
import yfinance as yf
import pandas as pd
from datetime import date, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar


# In[2]:


# Function pulling date from Yahoo Finance 

def fetch_save_data(ticker, start_date, end_date, file_name):
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Sort the DataFrame by date in descending order
    data_sorted = data.sort_index(ascending=False)
    
    # Print the head of the sorted data
    print(data_sorted.head())
    
    data.to_csv(file_name)


# In[3]:


# Check function works:

end_date = datetime.today().strftime('%Y-%m-%d')
fetch_save_data("INTC", start_date="1990-01-01", end_date=end_date, file_name="intel_data.csv")


# In[4]:


# Loading Data in order to scale it from -1 to 1

data = pd.read_csv('intel_data.csv', parse_dates=True, index_col='Date')


# In[5]:


# Scaling the Data from -1 to 1
# Question: How much does scaling matter when almost all variables are based in dollars? Volume is not on the same scale, but everything else is.

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
print(scaled_data.head())


# In[13]:


# Reads the file
# Tells pd to parse any columns that look like dates into DateTime objects
# Specify that Date is the index for the df
data = pd.read_csv('intel_data.csv', parse_dates=True, index_col='Date')

# Create previous time steps that include the last 5 days data
lags = 5
# Loop to create lagged features ranging from 1 to 5
for i in range(1, lags + 1):
    data[f'lag_{i}'] = data['Close'].shift(i)

# Drop rows with missing values
data = data.dropna()

# Dropping all values that aren't independent variables
# Assigning 'Close' as the dependent variable
X = data.drop(['Close', 'Adj Close'], axis=1)  # Using all columns as features except Close and Adj Close
y = data['Close']

# Split data and assign 20% of the data for testing and 80% for training
# Make sure the data is not being randomly shuffled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initializing the regression with the assigned training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict what the X_test value is
y_pred = model.predict(X_test)

# Evaluating the MSE between the true (y_yest) values and predicted (y_pred) values
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[11]:


# 1. Get the latest `lags` closing prices
latest_closing_prices = data['Close'][-lags:].values

# 2. Construct a feature vector for the next day
next_day_data = data.iloc[-1].drop(['Close', 'Adj Close']).copy()

for i, price in enumerate(reversed(latest_closing_prices)):
    next_day_data[f'lag_{i+1}'] = price

# Convert to DataFrame to maintain feature names1
next_day_features = pd.DataFrame([next_day_data.values], columns=next_day_data.index)

# 3. Predict the closing price using your trained model
predicted_next_day_close = model.predict(next_day_features)

print(f"Predicted closing price for the next trading day: ${predicted_next_day_close[0]:.2f}")



# In[20]:


# Function to append a csv with the prediction for the next trading day in the "prediction" Column
# If the csv at the file location doesn't exist, the function create's a file at that location

def save_prediction_to_csv(prediction):

    file_path = "/Users/nate/Desktop/Prediction Dashboard/INTEL_Predictions.csv"
    
    # Finding tomorrow's date
    tomorrow = date.today() + timedelta(days=1)
    
    # If the CSV file exists, read it. Otherwise, initialize a new DataFrame.
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col='Date')
    else:
        df = pd.DataFrame(columns=['Prediction', 'Actual Close'])
    
    # Save the prediction for tomorrow's date
    df.loc[tomorrow.strftime('%Y-%m-%d')] = [prediction, None]
    
    # Append the CSV file
    df.to_csv(file_path)

def update_actual_close(ticker, days_back=7):
    file_path = "/Users/nate/Desktop/Prediction Dashboard/INTEL_Predictions.csv"
    
    # Get today's date
    today = date.today().strftime('%Y-%m-%d')
    
    # Fetch the past `days_back` days of data from Yahoo Finance
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # If data for today is available
    if today in data.index:
        actual_close = data.loc[today, 'Close']
        
        # Read the existing CSV
        df = pd.read_csv(file_path, index_col='Date')
        
        # Update the 'Actual Close' column for today's date
        if today in df.index:
            df.at[today, 'Actual Close'] = actual_close
            
            # Write back to the CSV file
            df.to_csv(file_path)

# Update prediction for tomorrow's close
save_prediction_to_csv(predicted_next_day_close[0])
# Update today's close price
update_actual_close('INTC')