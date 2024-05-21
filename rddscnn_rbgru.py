# Importing necessary libraries
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, SeparableConv1D, Bidirectional, MaxPooling1D, AveragePooling1D, Input, Add, Dense, GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import forecast_evaluation

# Define a function for the residual dilated depthwise separable CNN block
def build_rddscnn_block(rddscnn_block_input):
    """
    Construct a residual dilated depthwise separable CNN block.

    Parameters:
        rddscnn_block_input (tensor): Input tensor to the block.

    Returns:
        tensor: Output tensor from the block.
    """
    # Middle branch of the block
    middle_1 = SeparableConv1D(filters=128, kernel_size=3, padding='same', dilation_rate=2, activation='relu', kernel_initializer='he_uniform')(rddscnn_block_input)
    middle_2 = SeparableConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, activation='relu', kernel_initializer='he_uniform')(middle_1)
    middle_3 = BatchNormalization()(middle_2)
    middle_4 = MaxPooling1D(padding='same')(middle_3)
    
    # Right branch of the block
    right_1 = BatchNormalization()(middle_1)
    right_2 = MaxPooling1D(padding='same')(right_1)
    right_3 = SeparableConv1D(filters=256, kernel_size=1, padding='same', dilation_rate=1, activation='relu', kernel_initializer='he_uniform')(right_2)
    
    # Left branch of the block
    left_1 = BatchNormalization()(middle_1)
    left_2 = AveragePooling1D(padding='same')(left_1)
    left_3 = SeparableConv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu', kernel_initializer='he_uniform')(left_2)
    
    # Combine the output of all branches
    rddscnn_block_output = Add()([right_3, middle_4, left_3])
  
    return rddscnn_block_output

# Define a function for the residual bidirectional GRU block
def build_rbgru_block(rbgru_block_input):
    """
    Construct a residual bidirectional GRU block.

    Parameters:
        rbgru_block_input (tensor): Input tensor to the block.

    Returns:
        tensor: Output tensor from the block.
    """
    bgru1 = Bidirectional(GRU(units=128, return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(rbgru_block_input)
    bgru2 = Bidirectional(GRU(units=128, return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(bgru1)
    concat_1 = Add()([bgru1, bgru2])
    
    bgru3 = Bidirectional(GRU(units=128, return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(concat_1)
    bgru1 = BatchNormalization()(bgru1)
    bgru2 = BatchNormalization()(bgru2)
    bgru3 = BatchNormalization()(bgru3)
    
    concat_2 = Add()([bgru1, bgru2, bgru3])
    rbgru_block_output = Bidirectional(GRU(units=128, return_sequences=False, activation='relu', kernel_initializer='he_uniform'))(concat_2)
    
    return rbgru_block_output

# Define a function for training and testing the hybrid model
def rddscnn_rbgru_model(X, y, n_steps_out, epochs=1, file_name='model_prediction.csv'):
    """
    Train and test the hybrid RDDSCNN-RBGRU model.

    Parameters:
        X (array-like): Input data.
        y (array-like): Target data.
        n_steps_out (int): Number of output time steps.
        epochs (int): Number of training epochs (default is 1).
        file_name (str): Name of the CSV file to save predictions (default is 'model_prediction.csv').

    Returns:
        None
    """
    # Obtain data shape
    batches, timesteps, features = X.shape

    # Reshape the input data if needed
    X = X.reshape(batches, timesteps, features)
   
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
  
    input_shape = x_train.shape[1:]
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Build the RDDSCNN block
    rddscnn_block_output = build_rddscnn_block(x)
    
    # Build the RBGRU block
    rbgru_block_output = build_rbgru_block(rddscnn_block_output)
    
    # Additional dense layer
    dense1 = Dense(units=128, activation='relu', kernel_initializer='he_uniform')(rbgru_block_output)
    
    # Output layer
    output = Dense(units=n_steps_out, activation='linear', kernel_initializer='he_uniform')(dense1)
    
    # Create the model
    rddscnn_rbgru_model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    rddscnn_rbgru_model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    print('[INFO]---|| *** Training the RDDSCNN-RBGRU Model...\n')
    rddscnn_rbgru_model.fit(x_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    print('[INFO]---|| *** RDDSCNN-RBGRU Model Trained!\n')
    
    # Save the model
    print('[INFO]---|| *** Saving the RDDSCNN-RBGRU Model...\n')
    rddscnn_rbgru_model.save('Models/rddscnn_rbgru_model.h5')
    print('[INFO]---|| *** RDDSCNN-RBGRU Model Saved!\n')
    
    # Test the model
    print('[INFO]---|| *** Testing the RDDSCNN-RBGRU Model...\n')  
    yhat = rddscnn_rbgru_model.predict(x_test, verbose=0)
    print('[INFO]---|| *** RDDSCNN-RBGRU Model Testing Completed!\n')

    # Save predictions to a CSV file
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': yhat.flatten()})
    df.to_csv(file_name, index=False)
    print("CSV file '{}' created successfully.".format(file_name))

    # Evaluate model predictions
    forecast_evaluation.evaluate_forecasts(y_test, yhat)
