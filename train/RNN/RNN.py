import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM - with GPU
from scipy.io import loadmat
import numpy as np

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

input_fNAME = 'Xih_3054x50k.mat'
output_fNAME = 'Yih_3054x50k.mat'

Xih = loadmat(input_fNAME)
inputDATA = Xih["Xih"].T
print(inputDATA[0].size )

Yih = loadmat(output_fNAME)
outputDATA = Yih["Yih"].T
print(outputDATA[0].size)

x_train = tf.keras.utils.normalize(inputDATA, axis=1)
y_train = tf.keras.utils.normalize(outputDATA, axis=1)

x_train = np.expand_dims(x_train, -1)

model = Sequential()
# IF you are running with a GPU, try out the CuDNNLSTM layer type instead LSTM (don't pass an activation, tanh is required)
model.add(LSTM(x_train[0].size, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(y_train[0].size, activation='softmax'))

opt = tf.keras.optimizers.Adam() #lr=0.001, decay=1e-6

model.compile(
    loss='mse',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(x_train,
          y_train,
          epochs=200,
          validation_data=0.2)