import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

'''Over All Variables'''
x_step = 100
y_step = 10
train_rate = 0.7
csv_path = "KRW-GMT.csv"
dividing = 10000
number_of_feature = 2

'''Get Dataset from csv'''
dataframe = pd.read_csv(csv_path)

volume = dataframe['volume'].values
price = dataframe['price'].values / dividing

def moving_average(data, window_size):
    # 주어진 윈도우 크기에 따라 가중치를 생성
    weights = np.repeat(1.0, window_size) / window_size

    # 이동 평균 계산
    moving_avg = np.convolve(data, weights, 'valid')

    return moving_avg


price = moving_average(price, 10)
volume = np.delete(volume, (1, 2, 3, 4, 5, 6, 7, 8, 9))

minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

volume = volume.reshape(-1, 1)
volume = minmax_scaler.fit_transform(volume)
price = price.reshape(-1, 1)

data = np.concatenate((price, volume), axis=1)

train = data[:round(len(data) * train_rate)]
test = data[round(len(data) * train_rate):]

price_train = price[:round(len(price) * train_rate)]
price_test = price[round(len(price) * train_rate):]

train_x = []
train_y = []

test_x = []
test_y = []

for i in range(0, len(train) - x_step - y_step + 1, (x_step // x_step)):
    train_x.append(train[i: i + x_step])
    train_y.append(price_train[i + x_step: i + x_step + y_step])

for i in range(0, len(test) - x_step - y_step + 1, (x_step // x_step)):
    test_x.append(test[i: i + x_step])
    test_y.append(price_test[i + x_step: i + x_step + y_step])

for i in range(len(train_x), len(train_x) - x_step, -1):
    if (train_x[i - 1].shape[0] != x_step) or (train_y[i - 1].shape[0] != y_step):
        del train_x[i - 1]
        del train_y[i - 1]

for i in range(len(test_x), len(test_x) - x_step, -1):
    if (test_x[i - 1].shape[0] != x_step) or (test_y[i - 1].shape[0] != y_step):
        del test_x[i - 1]
        del test_y[i - 1]

train_x = np.array(train_x)
train_y = np.array(train_y).reshape(-1, y_step)
test_x = np.array(test_x)
test_y = np.array(test_y).reshape(-1, y_step)

'''minmax scaling in each timestep'''
'''hadamard product in each timestep'''

'''minmax'''
for i in range(len(train_x)):
    train_x[i][:, 0] = minmax_scaler.fit_transform(train_x[i][:, 0].reshape(-1, 1)).reshape(x_step)
    train_y[i] = minmax_scaler.transform(train_y[i].reshape(-1, 1)).reshape(y_step)

for i in range(len(test_x)):
    test_x[i][:, 0] = minmax_scaler.fit_transform(test_x[i][:, 0].reshape(-1, 1)).reshape(x_step)
    test_y[i] = minmax_scaler.transform(test_y[i].reshape(-1, 1)).reshape(y_step)

model = Sequential()

model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Bidirectional(LSTM(units=100)))
model.add(Dense(y_step))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(train_x, train_y, epochs=10, batch_size=128)

prediction = model.predict(test_x)

plt.figure(figsize=(10, 10))
plt.title("GMT")
for i in range(1, 13):
    plt.subplot(4, 3, i)
    plt.plot(np.append([np.nan] * x_step, prediction[i * 500]))
    plt.plot(np.append(test_x[i * 500][:, 0], test_y[i * 500]))
    # plt.plot(prediction[i*500], label='Prediction')
    # plt.plot(test_y[i*500], label='Actual')

plt.legend(['prediction', 'actual'])
plt.show()

model.save('model0120/model.h5')
