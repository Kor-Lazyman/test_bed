import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Multiply, Flatten
# Note: Attention layer is available from Keras 2.4.0
from keras.layers import Attention
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('simulation_data.csv')  # Replace with your file path
# Extracting columns 1 to 10 (0-based indexing) as input data
input_data = data.iloc[:, 1:11].values
# Extracting columns 2 to 7 as output data
output_data = data.iloc[:, 2:8].values

# Normalize the data
scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()
input_data_normalized = scaler_input.fit_transform(input_data)
output_data_normalized = scaler_output.fit_transform(output_data)


# Function to create sequences of data for LSTM model training
def create_sequences(input_data, output_data, time_steps=5, forecast_horizon=1):
    X, y = [], []
    # Looping through the data to create sequences
    for i in range(len(input_data) - time_steps - forecast_horizon + 1):
        X.append(input_data[i:(i + time_steps)])
        y.append(output_data[i + time_steps + forecast_horizon - 1])
    return np.array(X), np.array(y)


# Creating sequences from the normalized data
X, y = create_sequences(input_data_normalized, output_data_normalized)
# Splitting the sequences into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)

# Function to build the LSTM model with an attention mechanism


def build_model(input_shape, output_size, lstm_units=50):
    # Defining the input layer
    inputs = Input(shape=input_shape)
    # LSTM layer with sequence and state outputs
    lstm_out, _, _ = LSTM(lstm_units, return_state=True,
                          return_sequences=True)(inputs)
    # Attention mechanism to weigh the importance of each time step in the sequence
    attention = Attention(use_scale=True)([lstm_out, lstm_out])
    # Multiplying the LSTM outputs with the attention weights
    merged = Multiply()([lstm_out, attention])
    # Flattening the merged output
    flattened = Flatten()(merged)
    # Dense (fully connected) layer with ReLU activation
    dense_layer = Dense(50, activation='relu')(flattened)
    # Output layer
    outputs = Dense(output_size)(dense_layer)

    # Compiling and returning the model
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Setting the input shape and output size based on training data
input_shape = (X_train.shape[1], X_train.shape[2])
output_size = y_train.shape[1]
# Building the model using the defined function
model = build_model(input_shape, output_size)

# Train model
# Training the model using the training data with 50 epochs and a batch size of 32
# 10% of the training data is used for validation
history = model.fit(X_train, y_train, epochs=50,
                    batch_size=32, validation_split=0.1)

# Plotting the training and validation losses across epochs
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training History')
plt.legend()
plt.show()

# Evaluating the model's performance on the test set
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Making predictions on the test set
y_pred = model.predict(X_test)
# Inverting the normalization on the true and predicted values to get them in their original scale
y_test_inv = scaler_output.inverse_transform(y_test)
y_pred_inv = scaler_output.inverse_transform(y_pred)

# Plotting the true vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[:, 0], label='True')
plt.plot(y_pred_inv[:, 0], label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Model Prediction')
plt.legend()
plt.show()
