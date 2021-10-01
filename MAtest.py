import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.io import loadmat
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.model_selection
import time

import autokeras as ak
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from scipy.io import loadmat
import time
import datetime
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input


np.random.seed(0)
m= 100
X= np.linspace(0,10,m).reshape(m,1)
y= X+ np.random.randn(m,1)

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_dim = 1))

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                loss='mean_absolute_error',
                metrics=[keras.metrics.Accuracy()] )

plt.scatter(X, y, label='data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()    
plt.show()

ynew= np.zeros((100,10))
for i in range(10):
    model.fit(X, y, epochs=i+1)
    ynew[:,i]= model.predict(X).T

    
plt.scatter(X, y, label='data')
plt.xlabel('X')
plt.ylabel('y')    
plt.scatter(X, ynew[:,0], label='Epoch #{}'.format(0))
plt.legend()    
plt.show()

plt.scatter(X, y, label='data')
plt.xlabel('X')
plt.ylabel('y')    
plt.scatter(X, ynew[:,0], label='Epoch #{}'.format(0))
plt.scatter(X, ynew[:,1], label='Epoch #{}'.format(1))
plt.legend()    
plt.show()

plt.scatter(X, y, label='data')
plt.xlabel('X')
plt.ylabel('y')
for i in range(10):  
    plt.scatter(X, ynew[:,i], label='Epoch #{}'.format(i))
plt.legend()    
plt.show()



