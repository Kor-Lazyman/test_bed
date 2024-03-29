# 첨부된 csv파일을 import해서 학습 및 테스트하는 파이썬 코드를 작성해줘.
# csv파일은 시간 순서대로 데이터를 저장하고 있고 1번 열은 시간(Day)이 흘러간 순서를 보여주고 있어.
# 2~6번 열을 가지고 7~11번 열을 예측하는 neural network 모델을 keras라이브러리를 이용해서 만들고 싶어.
# 특히, 7~11번 열은 시계열 데이터이기 때문에 이를 활용해서 다음 시점의 데이터를 예측하는데 도움이 될 수 있어.
# 학습 과정을 visualization해주고, test결과도 보여줘.
# 학습이 끝난 neural network모델은 다른곳에서 재사용 가능하도록 export해줘.
# 코드에 주석을 영어로 자세하게 넣어줘.

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# Load the dataset
# Replace 'attached_csv_file.csv' with the actual path if different
data = pd.read_csv('simulation_data.csv')

# Split data columns into inputs and targets
X = data.iloc[:, 1:6].values  # Input columns (2-6)
y = data.iloc[:, 6:11].values  # Target columns (7-11)

# Split the dataset into train and test sets
# For time series data, the typical train/test split is not randomized.
# We'll use the first 80% of data for training and the remaining 20% for testing.
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=5))  # Input layer
model.add(Dense(64, activation='relu'))  # Hidden layer
model.add(Dense(5))  # Output layer for 5 target columns

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_test, y_test), batch_size=32)

# Plot the training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model performance on test data
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# Save the model for future use
model.save("trained_time_series_model.h5")
