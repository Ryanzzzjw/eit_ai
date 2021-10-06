### this code is called by TRAIN.py

### using Tensorboard: open CMD, where logs file or this code is, and type " tensorboard --logdir=logs/ " or " py -m tensorboard.main --logdir=logs/ " or etc.
### then it can be seen http://H-PC:600 or http://localhost:6006/ or etc., copy it and paste it online to see training graphs live or already executed
### CMD must be opened all the time, when is wanted to see graphs. 

import autokeras as ak
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from scipy.io import loadmat
import time
import datetime

def training_data_loading(input_PATH, output_PATH):

    Xih = loadmat(input_PATH)
    inputDATA = Xih["Xih"].T
    print(inputDATA[0].size )

    Yih = loadmat(output_PATH)
    outputDATA = Yih["Yih"].T
    print(outputDATA[0].size)

    return inputDATA, outputDATA

def ML_KERAS(input, output, nameMODEL):
    start_time = time.time()
    NAME = "Model_Keras{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    print('Data normalization...')
    x_train = tf.keras.utils.normalize(input, axis=1)
    y_train = tf.keras.utils.normalize(output, axis=1)
    #y_train = output
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(x_train[0].size, input_dim = x_train[0].size, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(y_train[0].size, activation=tf.nn.sigmoid)) 

    model.output_shape
    model.compile(optimizer='adam',
                loss='binary_crossentropy',   ##'binary_crossentropy' or 'categorical_crossentropy' 
                metrics=['accuracy'])

    print(model.summary())

    print(x_train.shape , y_train.shape )
    print("Network compilation time = ", str(datetime.timedelta(time.time() - start_time)), 's')
    start_time = time.time()
    print('Strting training...')

    model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=[tensorboard])
    print('Training DONE!')
    print("Training time = ", str(datetime.timedelta(time.time() - start_time)), 's')
    
    model.save(nameMODEL + '.model')


def ML_autoKERAS(input, output, nameMODEL):
    start_time = time.time()
    NAME = "Model_autoKeras{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    print('Data normalization...')
    x_train = tf.keras.utils.normalize(input, axis=1)
    y_train = tf.keras.utils.normalize(output, axis=1)

    #x_train, x_test, y_train, y_test = model_selection.train_test_split(Xih,Yih, test_size=0.2)
    print('Strting training...')
    reg = ak.StructuredDataRegressor(max_trials = 10, overwrite=True)
    reg.fit(x_train, y_train, validation_split=0.20, epochs = 500, callbacks=[tensorboard]) #,epochs = 100

    print("Training time = ", str(datetime.timedelta(time.time() - start_time)), 's')
    print('Training DONE!')

    model = reg.export_model()
             
    print(model.summary())

    try:
        model.save(nameMODEL, save_format="tf")
    except Exception:
        model.save(nameMODEL + ".h5") 

def ML_optimization_TB(inputDATA, outputDATA, dense_layers, layer_sizes):
    start_time = time.time()
    x_train = tf.keras.utils.normalize(inputDATA, axis=1)
    y_train = tf.keras.utils.normalize(outputDATA, axis=1)

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Dense(x_train[0].size, input_dim = x_train[0].size, activation=tf.nn.relu))

            for _ in range(dense_layer):
                model.add(tf.keras.layers.Dense(layer_size, activation=tf.nn.relu))
            
            model.add(tf.keras.layers.Dense(y_train[0].size, activation=tf.nn.sigmoid)) 

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.output_shape
            model.compile(optimizer='adam',
                        loss='categorical_crossentropy',   #'binary_crossentropy' or 'categorical_crossentropy'
                        metrics=['accuracy'])
            print('Strting training...', NAME)
            model.fit(x_train, y_train, epochs=250, validation_split=0.2, callbacks=[tensorboard])

            print("Training time = ", str(datetime.timedelta(time.time() - start_time)), 's')
            print('Training DONE!')
