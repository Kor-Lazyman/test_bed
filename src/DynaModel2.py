from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# 데이터 로딩
data = pd.read_csv('simulation_data.csv')

# 2~6번 열을 입력값으로, 7~11번 열을 출력값으로 설정
X = data.iloc[:, 1:6].values
y = data.iloc[:, 6:11].values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=5))
model.add(Dense(64, activation='relu'))
model.add(Dense(5))

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_test, y_test))

# 학습 결과 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 테스트
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# 모델 저장
model.save("trained_model.h5")
