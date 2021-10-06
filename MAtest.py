# import numpy as np
# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.tri as mtri
# from scipy.io import loadmat
# # from tensorflow.compat.v1 import ConfigProto
# # from tensorflow.compat.v1 import InteractiveSession
# import tensorflow as tf
# import tensorflow.keras as keras
# import sklearn.model_selection
# import time

# import autokeras as ak
# import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard
# from scipy.io import loadmat
# import time
# import datetime
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input


# np.random.seed(0)
# m= 100
# X= np.linspace(0,10,m).reshape(m,1)
# y= 0.000000000000001*X+ np.random.randn(m,1)*0.00000000000000001+1




# from sklearn.linear_model import LinearRegression
# reg = LinearRegression().fit(X, y)
# print('R2:', reg.score(X, y))
# print(':', reg.coef_)
# print('score:', reg.intercept_)



# y_p= reg.predict(X)
# plt.scatter(X, y, label='data')
# plt.scatter(X, y_p, label='pre')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend() 
# plt.show()

# model = keras.models.Sequential()
# model.add(keras.layers.Dense(1, input_dim = 1))

# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.02),
#                 loss='mean_absolute_error',
#                 metrics=[keras.metrics.Accuracy()] )

# plt.scatter(X, y, label='data')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()    
# plt.show()

# ynew= np.zeros((100,10))
# for i in range(10):
#     model.fit(X, y, epochs=i+1)
#     ynew[:,i]= model.predict(X).T

    
# plt.scatter(X, y, label='data')
# plt.xlabel('X')
# plt.ylabel('y')    
# plt.scatter(X, ynew[:,0], label='Epoch #{}'.format(0))
# plt.legend()    
# plt.show()

# plt.scatter(X, y, label='data')
# plt.xlabel('X')
# plt.ylabel('y')    
# plt.scatter(X, ynew[:,0], label='Epoch #{}'.format(0))
# plt.scatter(X, ynew[:,1], label='Epoch #{}'.format(1))
# plt.legend()    
# plt.show()

# plt.scatter(X, y, label='data')
# plt.xlabel('X')
# plt.ylabel('y')
# for i in range(10):  
#     plt.scatter(X, ynew[:,i], label='Epoch #{}'.format(i))
# plt.legend()    
# plt.show()

if __name__ == "__main__":
    a='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/out'

    b='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/data/hjjjh/jkjkjk.g'
    import os
    print(os.path.relpath(b, start=a))

