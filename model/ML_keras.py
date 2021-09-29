import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.io import loadmat
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.model_selection

def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.

    # model = tf.keras.models.Sequential()
    # #model.add(tf.keras.layers.Flatten(input_shape=(Xih[1].shape,)))
    # model.add(tf.keras.layers.Dense(256, input_dim = 256, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(990, activation=tf.nn.sigmoid)) 
    # model.output_shape
    # model.compile(optimizer='adam',
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])



    inputs = keras.Input(shape=(256,))
    x = keras.layers.Dense(512, activation="relu")(inputs)
    x = keras.layers.Dense(1024, activation="relu")(x)
    outputs = keras.layers.Dense(990, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.Accuracy()],
    )
    model.summary()
    return model


def get_dataset(batch_size = 32, num_val_samples = 5000, test_size= 0.20, val_size=0.20):

    if test_size+val_size>=0.8:
        test_size= 0.2
        val_size=0.2


    # Load data
    Xih = loadmat(r"Xih_990x50k.mat") # <class 'numpy.ndarray'> 
    Xih = Xih["Xih"].T

    Yih = loadmat(r"Yih_990x50k.mat")
    Yih = Yih["Yih"].T

    print(np.shape(Xih) ,np.shape(Yih))

    X = tf.keras.utils.normalize(Xih, axis=0)

    #x_test = tf.keras.utils.normalize(x_test, axis=1)
    Y = Yih
    
    # make the 
    x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_size)
    print('nb samples : ', len(x_tmp), len(x_test))
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp, test_size=val_size)
    print('nb samples : ', len(x_train), len(x_val))
    #x_train = x_train.map(x_train)
    input_len= x_train[0].size
    output_len= y_train[0].size

    print('in ou : ', input_len, output_len)
    print('nb samples : ', len(X))

    # Preprocess the data (these are Numpy arrays)
    # x_train = X_train.astype("float32")
    # x_test = X_test.astype("float32")
    # y_train = y_train.astype("float32")
    # y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
  
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

# strategy = tf.distribute.get_strategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



# with strategy.scope():
#     model = tf.keras.models.Sequential()
#     #model.add(tf.keras.layers.Flatten(input_shape=(Xih[1].shape,)))
#     model.add(tf.keras.layers.Dense(input_len, input_dim = input_len, activation=tf.nn.relu))
#     model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
#     model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
#     model.add(tf.keras.layers.Dense(output_len, activation=tf.nn.sigmoid)) 

#     model.output_shape
#     model.compile(optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])

# model.summary()              
# print(x_train.shape , y_train.shape )

# model.fit(x_train, y_train, epochs=700, validation_split=0.2)
# #x_train = x_train.reshape(-1, 256)

# model.save('test_keras.model')

# predictions = model.predict(x_train)
# Yih[0] = predictions[0]


# # Create triangulation.
# tr_data = loadmat(r"tr_data_990el.mat")
# xy = np.array(tr_data["nodes"])
# triangles = np.array(tr_data["tr_points"])

# print(xy)

# x, y = xy.T

# print(x,y)

# print(triangles)
# triang = mtri.Triangulation(x, y)

# # Interpolate to regularly-spaced quad grid.
# conduct_elem = Yih[0]


# m = np.array(conduct_elem).flatten()
# n = np.array(triangles, dtype = float).flatten()

# print(m,n)

# z = [0 for a in range(np.size(x))]
# print(np.size(z))

# list = []

# for i in range (len(n)):
#     idx = n[int(i)] 
#     idx =int(idx)
#     #print(idx)
#     if idx not in list: 
#         j= int(i/3)
#         z[idx-1]=m[j]
#         list.append(idx)


# # Plot the triangulation.
# plt.plot()
# plt.tricontourf(triang, z)
# plt.triplot(triang, '-', alpha=.5)
# plt.title('Triangular grid')

# plt.tight_layout()
# plt.colorbar()

# plt.show()

if __name__ == "__main__":

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    train_dataset, val_dataset, test_dataset = get_dataset()
    
    print((train_dataset),len(val_dataset),len(test_dataset))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = get_compiled_model()

    # Train the model on all available devices.
    
    model.fit(train_dataset, epochs=10, validation_data=val_dataset)

    # Test the model on all available devices.
    model.evaluate(test_dataset)
