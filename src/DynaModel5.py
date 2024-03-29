import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Concatenate, Attention
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# Importing the CSV data
data = pd.read_csv('simulation_data.csv')

# Extract features and time-series data
features_data = data.iloc[:, [1] + list(range(3, 6))].values
time_series_data = data.iloc[:, 6:11].values

# Create shifted version of time series data for inputs
shifted_time_series_data = np.vstack(
    (np.zeros((1, time_series_data.shape[1])), time_series_data[:-1]))

# Concatenate features and shifted time series data
input_data = np.hstack((features_data, shifted_time_series_data))

# Create target data
targets = np.hstack((data.iloc[:, 2:3].values, time_series_data))

# Reshaping data for LSTM layer
input_data = np.reshape(input_data, (-1, 5, 7))

# Build model

# Define feature inputs
features_input = Input(shape=(5, 7))

# LSTM layers
lstm_out, state_h, state_c = LSTM(
    50, return_sequences=False, return_state=True)(features_input)

# Attention layers
attention = Attention()([lstm_out, features_input])
concat = Concatenate(axis=-1)([lstm_out, attention])

# Dense layers
dense_out = Dense(6, activation='linear')(concat)

# Compile the model
model = Model(features_input, dense_out)
model.compile(optimizer='adam', loss='mean_squared_error')

# Summarize the model
model.summary()

# Training the model
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('model.h5', save_best_only=True)
]

history = model.fit(input_data, targets, epochs=100,
                    batch_size=32, validation_split=0.2, callbacks=callbacks)

# Visualizing training process
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Load best model
best_model = load_model('model.h5')

# Making predictions on new data (you can replace test_data with your actual test data)
# test_data should have the same format as input_data
# predictions = best_model.predict(test_data)

# Save the trained model for later use
model.save("trained_model.h5")
