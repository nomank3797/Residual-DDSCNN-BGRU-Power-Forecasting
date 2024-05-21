import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import nan, isnan
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Function to fill missing values with the value at the same time one day ago
def fill_missing(values):
    """
    Fill missing values in a dataset with values from the same time one day ago.

    Parameters:
        values (numpy.ndarray): Input dataset.

    Returns:
        None
    """
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]

# Function to apply exponential smoothing
def holt_winters_smoothing(data, trend=None, seasonal=None, smoothing_level=0.5, smoothing_trend=0.5):
    """
    Apply Holt-Winters exponential smoothing to the dataset.

    Parameters:
        data (numpy.ndarray): Input dataset.
        trend (str): Type of trend component (default is None).
        seasonal (str): Type of seasonal component (default is None).
        smoothing_level (float): Smoothing parameter for level (default is 0.5).
        smoothing_slope (float): Smoothing parameter for slope (default is 0.5).

    Returns:
        numpy.ndarray: Smoothed dataset.
    """
    smoothed_data = np.zeros_like(data)
    for i in range(data.shape[1]):  # Iterate over each feature
        model = ExponentialSmoothing(data[:, i], trend=trend, seasonal=seasonal)
        fit_model = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend)
        smoothed_data[:, i] = fit_model.fittedvalues
    return smoothed_data

# Function to normalize data
def normalize_data(values):
    """
    Normalize the dataset using Min-Max scaling.

    Parameters:
        values (numpy.ndarray): Input dataset.

    Returns:
        numpy.ndarray: Normalized dataset.
    """
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler on data
    scaler.fit(values)
    # Apply transform
    normalized = scaler.transform(values)
    return normalized

# Function to clean up data
def clean_data(data, frequency='M'):
    """
    Clean up the dataset by filling missing values, resampling, smoothing, and normalizing.

    Parameters:
        data (pandas.DataFrame): Input dataset.
        frequency (str): Resampling frequency (default is 'M' for monthly).

    Returns:
        numpy.ndarray: Cleaned and preprocessed dataset.
    """
    # Mark all missing values
    data.replace('?', nan, inplace=True)
    # Make dataset numeric
    data = data.astype('float32')
    # Fill missing values
    fill_missing(data.values)
    # Resample data
    resample_groups = data.resample(frequency)
    resample_data = resample_groups.mean()
    # Exponential smoothing
    smoothed_data = holt_winters_smoothing(resample_data.values)
    # Normalize data
    normalized_data = normalize_data(smoothed_data)
    return normalized_data

# Function to split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    """
    Split a multivariate sequence into input/output samples.

    Parameters:
        sequences (numpy.ndarray): Input sequence.
        n_steps_in (int): Number of input time steps.
        n_steps_out (int): Number of output time steps.

    Returns:
        tuple: Input and output sequences.
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
