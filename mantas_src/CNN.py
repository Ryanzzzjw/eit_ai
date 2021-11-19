import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from scipy.io import loadmat
import numpy as np

physical_devs = tf.config.experimental.list_physical_devices('GPU')
print("nb GPU availables: ", physical_devs)
tf.config.experimental.set_memory_growth(physical_devs, True)


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
print(x_train.shape[0:])


model = Sequential()
model.add(Conv1D(256, 16, input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling1D(9))

model.add(Conv1D(256, 9 ))
model.add(Activation('relu'))
model.add(MaxPooling1D(4))

model.add(Flatten()) 
model.add(Dense(128))

model.add(Dense(y_train[0].size))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=200, validation_split=0.2)