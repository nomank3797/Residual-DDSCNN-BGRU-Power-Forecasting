import numpy as np
import pandas as pd
import data_processing
import rddscnn_rbgru

# Load data from the CSV file
# Assuming 'household_power_consumption.txt' is in the 'raw data' directory
dataset = pd.read_csv('raw data/household_power_consumption.txt', sep=';', header=0, low_memory=False,
                      infer_datetime_format=True, parse_dates={'datetime': [0, 1]}, index_col=['datetime'])

# Choose data frequency (e.g., 'H' for hourly)
frequency = 'H'

# Define the number of time steps for input and output
n_steps_in, n_steps_out = 1, 1

# Number of epochs for training
epochs = 100

# Define the filename for saving predicted values
file_name = 'hourly_actual_predicted_values.csv'

# Clean the raw data
cleaned_data = data_processing.clean_data(dataset, frequency)

# Convert data into input/output sequences
X, y = data_processing.split_sequences(cleaned_data, n_steps_in, n_steps_out)

# Train and test the model
rddscnn_rbgru.rddscnn_rbgru_model(X, y, n_steps_out, epochs, file_name)
