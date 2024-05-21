from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from math import sqrt

# Evaluate one or more forecasts against expected values
def evaluate_forecasts(actual, predicted):
    """
    Evaluate one or more forecasts against the expected values.

    Args:
        actual (array-like): Actual values.
        predicted (array-like): Predicted values.

    Returns:
        None
    """
    # Calculate mean squared error (MSE)
    mse = mean_squared_error(actual, predicted)

    # Calculate root mean squared error (RMSE)
    rmse = sqrt(mse)

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(actual, predicted)

    # Print evaluation metrics
    print("MSE: {:.4f}".format(mse))
    print("RMSE: {:.4f}".format(rmse))
    print("MAE: {:.4f}".format(mae))
