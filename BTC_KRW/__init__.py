from keras import layers, Model, Input, Sequential, optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

x_step = 200
y_step = 40
train_rate = 0.95
csv_path = "KRW-BTC.csv"
dividing = 10000

scaler = MinMaxScaler()


df = pd.read_csv('KRW-BTC.csv')
print(df.head())
print(df.dtypes)

price = np.array(df['close'].values)
volume = np.array(df['volume'].values)

price = np.reshape(np.array(price), (-1, 1))
# volume = np.reshape(np.array(volume), (-1, x_step+y_step))
price = scaler.fit_transform(price)
price = price.flatten()


# volume = np.reshape(np.array(volume), (len(volume), 1))
# volume = scaler.fit_transform(volume)
# volume = volume.flatten()

price_train = price[:round(len(price)*train_rate)].tolist()
volume_train = price[:round(len(volume)*train_rate)].tolist()
price_test = price[round(len(price)*train_rate):].tolist()
volume_test = price[round(len(volume)*train_rate):].tolist()

volume_x = []
volume_y = []
price_x = []
price_y = []
for i in range(0, len(price_train) - x_step - y_step + 1, 5) :
    volume_x.append(volume_train[i:i+x_step])
    volume_y.append(volume_train[i+x_step:i+x_step+y_step])
    # volume_y.append(volume_train[i:i+x_step+y_step])
    price_x.append(price_train[i:i+x_step])
    price_y.append(price_train[i+x_step:i+x_step+y_step])
    # price_y.append(price_train[i:i+x_step+y_step])

# plt.plot(price_x[1], label='1')
# plt.plot(price_x[2], label='2')
# plt.plot(price_x[3], label='3')
# plt.show()

randomize = np.arange(len(price_x))
np.random.shuffle(randomize)

volume_x = np.array(volume_x)[randomize]
volume_y = np.array(volume_y)[randomize]
price_x = np.array(price_x)[randomize]
price_y = np.array(price_y)[randomize]

print('loop - done')

volume_x = np.reshape(volume_x, (len(volume_x), len(volume_x[1]), 1))
volume_y = np.reshape(volume_y, (len(volume_y), len(volume_y[1]), 1))
price_x = np.reshape(price_x, (len(price_x), len(price_x[1]), 1))
price_y = np.reshape(price_y, (len(price_y), len(price_y[1]), 1))

print(price_x.shape)
print(price_y.shape)

# train_data = np.stack([price_x, volume_x], axis=2)
# train_data = np.reshape(
#     train_data,
#     (len(train_data), len(train_data[1]), 2)
# )


model = Sequential()
# model.add(layers.LSTM(units=16))

# model.add(layers.Input(shape=(x_step, 1)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(200))
# model.add(layers.LSTM(units=2))
# model.add(layers.Conv1D(
#     filters=6,
#     kernel_size=1,
#     padding="same",
#     strides=1,
#     activation='relu',
# ))
# model.add(layers.Conv1D(
#     filters=6,
#     kernel_size=1,
#     padding="same",
#     strides=1,
#     activation='relu',
# ))
# model.add(layers.Conv1D(
#     filters=6,
#     kernel_size=1,
#     padding="same",
#     strides=1,
#     activation='relu',
# ))
model.add(layers.Flatten())
model.add(layers.Dense(200))
# model.add(layers.Dense(190))
model.add(layers.Dense(180))
# model.add(layers.Dense(170))
model.add(layers.Dense(160))
# model.add(layers.Dense(150))
model.add(layers.Dense(140))
# model.add(layers.Dense(130))
model.add(layers.Dense(120))
# model.add(layers.Dense(110))
model.add(layers.Dense(100))
model.add(layers.Dense(80 ))
model.add(layers.Dense(60 ))
model.add(layers.Dense(40 ))
model.add(layers.Dense(20 ))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(y_step))

# layer1 = layers.Dense(12, activation='relu')


# model.add(layers.LSTM(units=2, input_shape=(x_step, 4)))
# model.add(layers.LSTM(units=12, return_sequences=True))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(units=128))
# model.add(layers.LSTM(units=24, return_sequences=True))
# model.add(layers.LSTM(units=6))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(units=y_step))

opt = optimizers.Adam(learning_rate=0.0000001)
model.compile(optimizer=opt, loss='mse')

print(model)

# print(np.reshape(price_x, (-1, 1, x_step)).shape)

model.fit(
  price_x,
  # np.reshape(
  #   np.stack([price_x, volume_x], axis=2),
  #   (-1, x_step, 2),
  # ),
  price_y,
  batch_size=5,
  epochs=100
)


# validation = np.stack([price_x[-2], volume_x[-2]], axis=2)
# validation = np.reshape(validation, (-1, x_step, 2))


volume_tx = []
volume_ty = []
price_tx = []
price_ty = []

'''testset dividing'''
'''all array has same length'''
for i in range(len(price_test) - x_step - y_step + 1):
    volume_tx.append(volume_test[i:i + x_step])
    volume_ty.append(volume_test[i + x_step:i + x_step + y_step])
    price_tx.append(price_test[i:i + x_step])
    price_ty.append(price_test[i + x_step:i + x_step + y_step])


volume_tx = np.array(volume_tx)
volume_ty = np.array(volume_ty)
price_tx = np.array(price_tx)
price_ty = np.array(price_ty)

volume_tx = np.reshape(volume_tx, (len(volume_tx), len(volume_tx[1])))
volume_ty = np.reshape(volume_ty, (len(volume_ty), len(volume_ty[1])))
price_tx = np.reshape(price_tx, (len(price_tx), len(price_tx[1])))
price_ty = np.reshape(price_ty, (len(price_ty), len(price_ty[1])))



# test = np.stack([volume_tx[-2], price_tx[-2]], axis=2)
# test = np.reshape(test, (-1,x_step, 4))
price_forecast = model.predict(
  price_tx,
  # np.reshape(
  #   np.stack([price_tx, volume_tx], axis=2),
  #   (-1, x_step, 2),
  # ),
)
print('forecast -- ')
print(price_forecast)
print(price_forecast.shape)
# hi = np.reshape(
#   price_forecast,
#   (y_step,)
# )
# zero = np.array([np.nan, np.nan , np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
# hi = np.append(zero, hi)

zero_pad = np.empty((len(price_tx[-2]),))
zero_pad[:] = np.nan
last = np.append(price_tx[-2], price_ty[-2])

plt.plot(last)
plt.plot(np.append(zero_pad, price_forecast[-2]))
plt.savefig('predict-dense-and-lstm.png')
plt.show()

model.save('model_0109/model.h5')