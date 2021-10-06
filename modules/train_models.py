### this code is called by TRAIN.py

### using Tensorboard: open CMD, where logs file or this code is, and type " tensorboard --logdir=logs/ " or " py -m tensorboard.main --logdir=logs/ " or etc.
### then it can be seen http://H-PC:600 or http://localhost:6006/ or etc., copy it and paste it online to see training graphs live or already executed
### CMD must be opened all the time, when is wanted to see graphs. 

from modules.dataset import EITDataset4ML
from modules.train_utils import *

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import tensorflow.keras as keras
import time
import autokeras as ak
from scipy.io import loadmat
import time
import os


class ModelGenerator(object):
    def __init__(self) -> None:
        super().__init__()
        self.model= []
        self.model_name= ''
        self.info= ''

    def select_model(self, train_inputs):

        train_inputs.model_func(train_inputs=train_inputs)

    def compile_model(  self,
                        optimizer=keras.optimizers.Adam(),
                        loss=keras.losses.CategoricalCrossentropy(),
                        metrics=[keras.metrics.Accuracy()],
                        train_inputs:TrainInputs=None):

        if train_inputs:
            optimizer=train_inputs.optimizer
            loss=train_inputs.loss
            metrics= train_inputs.metrics
        
        if self.model_name and self.model_name.find('autokeras')==-1:
            self.info= self.info +''  # to do
            self.model.compile( optimizer=optimizer,
                                loss=loss,
                                metrics=metrics)
            

    def save_model(self, path ):
        
        from contextlib import redirect_stdout

        with open(os.path.join(path, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        if self.model_name.find('autokeras')==-1:
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
                dataset:EITDataset4ML,
                epochs=100,
                steps_per_epoch=None,
                validation_steps=None,
                callbacks=None,
                train_inputs:TrainInputs=None):

        if train_inputs:
            epochs=train_inputs.epoch
            steps_per_epoch=train_inputs._steps_per_epoch
            validation_steps=train_inputs._validation_steps
            callbacks=train_inputs.callbacks

        start_time = time.time()
        if dataset.use_tf_dataset:
            self.model.fit( dataset.train,
                            epochs=epochs,
                            validation_data=dataset.val,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            callbacks=callbacks)
        else:
            self.model.fit( dataset.train.features,
                            dataset.train.labels,
                            epochs=epochs,
                            validation_data=(dataset.val.features, dataset.val.labels),
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            callbacks=callbacks)

        if self.model_name.find('autokeras')>=0:
            self.model = self.model.export_model()

        print('\n Training lasted: {}s'.format(time.time()- start_time),)

    def std_keras(  self, 
                    input_size=10,
                    output_size=10,
                    train_inputs:TrainInputs=None):

        if train_inputs:
            input_size=train_inputs.input_size
            output_size=train_inputs.output_size
      

        self.model_name = "std_keras"
        self.info= '' # to do
        #y_train = output
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(input_size, input_dim = input_size, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(512, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(512, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(output_size, activation=tf.nn.sigmoid)) 

        return self.model

    def std_keras2(  self, 
                    input_size=10,
                    output_size=10,
                    train_inputs:TrainInputs=None):

        if train_inputs:
            input_size=train_inputs.input_size
            output_size=train_inputs.output_size

        self.model_name = "std_keras2"
        self.info= '' # to do
        #y_train = output
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(input_size, input_dim = input_size, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(512, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(output_size, activation=tf.nn.sigmoid)) 

        return self.model

    def std_autokeras(self,
                    max_trials= 10,
                    train_inputs:TrainInputs=None):
        if train_inputs:
            max_trials=train_inputs.max_trials_autokeras
        
        self.model_name = "std_autokeras"
        self.info= '' # to do
        
        self.model = ak.StructuredDataRegressor(max_trials = max_trials, overwrite=True)

        return self.model


if __name__ == "__main__":
    gen= ModelGenerator()
    gen.select_model( model_func=gen.std_keras)
    print(type(gen.model))
    gen.compile_model()
    model= gen.model
    print(model.summary())
    
