### this code is called by TRAIN.py

### using Tensorboard: open CMD, where logs file or this code is, and type " tensorboard --logdir=logs/ " or " py -m tensorboard.main --logdir=logs/ " or etc.
### then it can be seen http://H-PC:600 or http://localhost:6006/ or etc., copy it and paste it online to see training graphs live or already executed
### CMD must be opened all the time, when is wanted to see graphs. 

from modules.dataset import EITDataset4ML
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
import os

class ModelGenerator():
    def __init__(self) -> None:
        self.model= []
        self.name= ''
        self.info= ''
    
    # def select_model(self, model_func, input_size=256, output_size=990, **args, **kwargs):
    
    #     self.model = model_func(input_size, output_size,**args, **kwargs)

    def compile_model(  self,
                        optimizer=keras.optimizers.Adam(),
                        loss=keras.losses.CategoricalCrossentropy(),
                        metrics=[keras.metrics.Accuracy()]):

        
        if self.name and self.name.find('autokeras')==-1:
            self.info= self.info +''  # to do
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics )

    def save_model(self, path ):
        
        from contextlib import redirect_stdout

        with open(os.path.join(path, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        if self.name.find('autokeras')==-1:
            self.model.save(os.path.join(path,'model.model'))
        else:
            try:
                self.model.save(os.path.join(path, 'model'), save_format="tf")
            except Exception:
                self.model.save(os.path.join(path, 'model.h5'))
        pass

    def load_model(self, path ):
        self.model = tf.keras.models.load_model(path, custom_objects=ak.CUSTOM_OBJECTS)
        

    
    def mk_fit(self,
                dataset=EITDataset4ML(),
                epochs=100,
                steps_per_epoch=None,
                validation_steps=None,
                callbacks=None):

        start_time = time.time()
        if dataset.use_tf_dataset:
            self.model.fit(dataset.train,
                            epochs=epochs,
                            validation_data=dataset.val,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            callbacks=callbacks)
        else:
            self.model.fit(dataset.train.features,
                            dataset.train.labels,
                            epochs=epochs,
                            validation_data=(dataset.val.features, dataset.val.labels),
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            callbacks=callbacks)
        if self.name.find('autokeras')>=0:
            self.model = self.model.export_model()

        print('\n Training lasted: {}s'.format(time.time()- start_time),)

    def std_keras(  self, 
                    input_size,
                    output_size):

        self.name = "std_keras"
        self.info= '' # to do
        #y_train = output
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(input_size, input_dim = input_size, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(512, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(512, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(output_size, activation=tf.nn.sigmoid)) 

        return self.model

    def std_keras2(  self, 
                    input_size,
                    output_size):

        self.name = "std_keras2"
        self.info= '' # to do
        #y_train = output
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(input_size, input_dim = input_size, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(512, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(output_size, activation=tf.nn.sigmoid)) 

        return self.model

    def std_autokeras(self, input_size, output_size, max_trials= 10):
        self.name = "std_autokeras"
        self.info= '' # to do
        
        self.model = ak.StructuredDataRegressor(max_trials = max_trials, overwrite=True)
        # reg.fit(x_train, y_train, validation_split=0.20, epochs = 500, callbacks=[tensorboard]) #,epochs = 100

        # print("Training time = ", str(datetime.timedelta(time.time() - start_time)), 's')
        # print('Training DONE!')

        # model = reg.export_model()
                
        # print(model.summary())

        # try:
        #     model.save(nameMODEL, save_format="tf")
        # except Exception:
        #     model.save(nameMODEL + ".h5") 
        return self.model

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


if __name__ == "__main__":
    gen= ModelGenerator()
    gen.select_model( model_func=gen.std_keras)
    print(type(gen.model))
    gen.compile_model()
    model= gen.model
    print(model.summary())
    
