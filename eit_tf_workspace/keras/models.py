

import os
from contextlib import redirect_stdout
from logging import error, getLogger
from typing import Any

import autokeras as ak
import eit_tf_workspace.constants as const
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from eit_tf_workspace.keras.const import (KERAS_LOSS,
                                          KERAS_MODEL_SAVE_FOLDERNAME,
                                          KERAS_OPTIMIZER, KerasLosses,
                                          KerasOptimizers)
from eit_tf_workspace.train_utils.dataset import Datasets
from eit_tf_workspace.train_utils.lists import KerasModels
from eit_tf_workspace.train_utils.metadata import MetaData
from eit_tf_workspace.train_utils.models import (MODEL_SUMMARY_FILENAME,
                                                 ModelNotDefinedError,
                                                 ModelNotPreparedError, Models,
                                                 WrongLearnRateError,
                                                 WrongLossError,
                                                 WrongMetricsError,
                                                 WrongOptimizerError)
from genericpath import isdir

logger = getLogger(__name__)


################################################################################
# Std Keras Model
################################################################################
class StdKerasModel(Models):

    def _define_model(self, metadata:MetaData)-> None:
        self.name = "std_keras"
        in_size=metadata.input_size
        out_size=metadata.output_size
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(in_size, input_dim = in_size))
        self.model.add(keras.layers.Activation(tf.nn.relu))
        self.model.add(keras.layers.Dense(512))
        self.model.add(keras.layers.Activation(tf.nn.relu))
        self.model.add(keras.layers.Dense(512))
        self.model.add(keras.layers.Activation(tf.nn.relu))
        self.model.add(keras.layers.Dense(out_size)) 
        self.model.add(keras.layers.Activation(tf.nn.sigmoid))
    
    def _get_specific_var(self, metadata:MetaData)-> None:        

        self.specific_var['optimizer']= get_keras_optimizer(metadata)
        self.specific_var['loss'] = get_keras_loss(metadata)
        if not isinstance(metadata.metrics ,list): #Could be better tested... TODO
            raise WrongMetricsError(f'Wrong metrics type: {metadata.metrics}') 
        self.specific_var['metrics']=metadata.metrics

    def _prepare_model(self)-> None:
        assert_keras_model_defined(self.model)
        self.model.compile( 
            optimizer=self.specific_var['optimizer'],
            loss=self.specific_var['loss'],
            metrics=self.specific_var['metrics'])

    def train(self, dataset:Datasets, metadata:MetaData)-> None:
        assert_keras_model_compiled(self.model)
        self.model.fit(
            x=dataset.get_X('train'),
            y=dataset.get_Y('train'),
            epochs=metadata.epoch,
            validation_data=(dataset.get_X('val'), dataset.get_Y('val')),
            steps_per_epoch=metadata._steps_per_epoch,
            validation_steps=metadata._validation_steps,
            callbacks=metadata.callbacks,
            batch_size=metadata.batch_size)

    def predict(
        self,
        X_pred:np.ndarray,
        metadata:MetaData,
        **kwargs)->np.ndarray:

        assert_keras_model_compiled(self.model)
        steps=metadata._test_steps
        if X_pred.shape[0]==1:
            steps= 1
        return self.model.predict(X_pred, steps=steps, **kwargs)

    def save(self, metadata:MetaData)-> str:
        assert_keras_model_compiled(self.model)
        return save_keras_model(self.model, dir_path=metadata.dir_path, save_summary=metadata.save_summary)

    def load(self, metadata:MetaData)-> None:
        self.model=load_keras_model(metadata)
        assert_keras_model_compiled(self.model)

################################################################################
# Std Autokeras
################################################################################
class StdAutokerasModel(Models):
    def _define_model(self, metadata:MetaData)-> None:
        self.name = "std_autokeras"
        self.model = ak.StructuredDataRegressor(
            max_trials = metadata.max_trials_autokeras, 
            overwrite=True, 
            directory=metadata.dir_path)
    def _get_specific_var(self, metadata:MetaData)-> None:
        """"""
    def _prepare_model(self)-> None:
        # assert_keras_model_defined(self.model)
        """ """
    def train(self, dataset:Datasets, metadata:MetaData)-> None: 
        # assert_keras_model_compiled(self.model)   
        self.model.fit(
            x=dataset.get_X('train'),
            y=dataset.get_Y('train'),
            epochs=metadata.epoch,
            validation_data=(dataset.get_X('val'), dataset.get_Y('val')),
            steps_per_epoch=metadata._steps_per_epoch,
            validation_steps=metadata._validation_steps,
            callbacks=metadata.callbacks,
            batch_size=metadata.batch_size)
        self.model=self.model.tuner.get_best_model()

    def predict(
        self,
        X_pred:np.ndarray,
        metadata:MetaData,
        **kwargs)->np.ndarray:

        # assert_keras_model_compiled(self.model)
        steps=metadata._test_steps
        if X_pred.shape[0]==1:
            steps= 1
        return self.model.predict(X_pred, steps=steps, **kwargs)

    def save(self, metadata:MetaData)-> str:
        # assert_keras_model_compiled(self.model)
        return save_keras_model(self.model, dir_path=metadata.dir_path, save_summary=metadata.save_summary)

    def load(self, metadata:MetaData)-> None:
        self.model=load_keras_model(metadata)

################################################################################
# Methods for keras / Autokeras
################################################################################

def assert_keras_model_defined(model:Any)->keras.models.Model:
    """allow to react if model not  defined

    Args:
        model (Any): [description]

    Raises:
        ModelNotDefinedError: [description]

    Returns:
        keras.models.Model: [description]
    """    
    if not isinstance(model, keras.models.Model):
        raise ModelNotDefinedError(f'Model has not been correctly defined: {model}')
    return model

def assert_keras_model_compiled(model:Any)->None:
    """allow to react if model not  defined

    Args:
        model (Any): [description]
    """    
    model:keras.models.Model=assert_keras_model_defined(model)
    try:
        model._assert_compile_was_called() #raise a RuntimeError if not compiled
    except RuntimeError as e:
        raise ModelNotPreparedError(f'Model need to be compiled first : ({e})')


def get_keras_optimizer(metadata:MetaData)-> keras.optimizers.Optimizer:

    if not metadata.optimizer:
        metadata.optimizer=list(KERAS_OPTIMIZER.keys())[0].value
    try:
        optimizer=KERAS_OPTIMIZER[KerasOptimizers(metadata.optimizer)]()
    except ValueError:
        raise WrongOptimizerError(f'Wrong optimizer type: {metadata.optimizer}')

    if metadata.learning_rate:
        if metadata.learning_rate >= 1.0:
            raise WrongLearnRateError(f'Wrong learning rate type (>= 1.0): {metadata.learning_rate}') 
        optimizer.learning_rate= metadata.learning_rate

    return optimizer

def get_keras_loss(metadata:MetaData)-> keras.losses.Loss:

    if not metadata.loss:
        metadata.loss=list(KERAS_LOSS.keys())[0].value
    try:
        loss=KERAS_LOSS[KerasLosses(metadata.loss)]()
    except ValueError:
        raise WrongLossError(f'Wrong loss type: {metadata.loss}')

    return loss

def save_keras_model(model:keras.Model, dir_path:str='', save_summary:bool=False)-> str:
    """Save a Keras model, additionnaly can be the summary of the model be saved"""
    if not isdir(dir_path):
        dir_path=os.getcwd()
    model_path=get_path_keras_model(dir_path)
    model.save(model_path)

    logger.info(f'Keras model saved in: {model_path}')
    
    if save_summary:
        summary_path= os.path.join(dir_path, MODEL_SUMMARY_FILENAME)
        with open(summary_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        logger.info(f'Keras model summary saved in: {summary_path}')
    
    return model_path

def load_keras_model(metadata:MetaData) -> keras.models.Model:
    """Load keras Model and return it if succesful if not """

    # if not isdir(dir_path):
    #     logger.info(f'Keras model loading - failed, wrong dir {dir_path}')
    #     return
    # model_path=get_path_keras_model(dir_path)

    model_path=get_path_keras_model(metadata.dir_path)
    if get_path_keras_model(metadata.dir_path,metadata.model_saving_path[1]) != model_path:
        logger.warning(f'The saved path in metadata "{metadata.model_saving_path}" is not the automatic one!')

    if not isdir(model_path):
        logger.info(f'Keras model loading - failed, {model_path} do not exist')
        return None
    try:
        model:keras.models.Model = keras.models.load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        logger.info(f'Keras model loaded: {model_path}')
        logger.info('Keras model summary:')
        model.summary()
        return model
    except BaseException as e: 
        logger.error(f'Loading of model from dir: {model_path} - Failed'\
                     f'\n({e})')
        return None

def get_path_keras_model(dir_path:str, default_filename:str=KERAS_MODEL_SAVE_FOLDERNAME )-> str:
    return os.path.join(dir_path, default_filename)

################################################################################
# Keras Models
################################################################################

KERAS_MODELS={
    KerasModels.StdKerasModel: StdKerasModel,
    KerasModels.StdAutokerasModel: StdAutokerasModel
}


if __name__ == "__main__":
    import logging

    from eit_tf_workspace.utils.log import change_level, main_log
    main_log()
    change_level(logging.DEBUG)

    """"""
    

    
