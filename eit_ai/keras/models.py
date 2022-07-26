

from abc import ABC, abstractmethod
import os
from contextlib import redirect_stdout
import logging
from typing import Any

import autokeras as ak
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from eit_ai.keras.const import (KERAS_LOSSES, KERAS_MODEL_SAVE_FOLDERNAME,
                                KERAS_OPTIMIZERS)
from eit_ai.train_utils.dataset import AiDatasetHandler
from eit_ai.train_utils.lists import (ListKerasLosses, ListKerasModelHandlers, ListKerasModels,
                                      ListKerasOptimizers, get_from_dict)
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.models import (MODEL_SUMMARY_FILENAME,
                                       ModelNotDefinedError,
                                       ModelNotPreparedError, AiModelHandler,
                                       WrongLearnRateError, WrongLossError,
                                       WrongMetricsError, WrongOptimizerError)
from genericpath import isdir

logger = logging.getLogger(__name__)



class TypicalKerasModelGenerator(ABC):
    """To Allow a easy definition of different 
    """    
    model:keras.models.Model=None
    name:str=None

    def __init__(self,metadata:MetaData) -> None:
        self._set_layers(metadata=metadata)

    @abstractmethod
    def _set_layers(self, metadata:MetaData)->None:
        """define the layers of the model and the name

        Args:
            metadata (MetaData): [description]

        """
    def get_name(self)->str:
        """Return the name of the model/network

        Returns:
            str: specific name of the model/network
        """        
        return self.name

    def get_model(self)->keras.models.Model:
        """Return the keras model/network

        Returns:
            keras.models.Model: the keras model/network
        """     
        return self.model

class StdKerasModel(TypicalKerasModelGenerator):
    """Define a Standard
    """    

    def _set_layers(self, metadata:MetaData)->None:
        self.name = "std_keras 2 dense layers 512 +relu +sigmoid"
        in_size=metadata.input_size
        out_size=metadata.output_size
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(in_size, input_dim = in_size))
        self.model.add(keras.layers.Activation(tf.nn.relu))
        self.model.add(keras.layers.Dense(1024))
        self.model.add(keras.layers.Activation(tf.nn.relu))
        self.model.add(keras.layers.Dense(128))
        self.model.add(keras.layers.Activation(tf.nn.relu))
        self.model.add(keras.layers.Dense(1024))
        self.model.add(keras.layers.Activation(tf.nn.relu))
        self.model.add(keras.layers.Dense(out_size)) 
        self.model.add(keras.layers.Activation(tf.nn.sigmoid))

class StdAutokerasModel(TypicalKerasModelGenerator):
    """Define a Standard
    """
    def _set_layers(self, metadata:MetaData)->None:
        self.name = "std_autokeras"
        self.model = ak.StructuredDataRegressor(
            max_trials = metadata.max_trials_autokeras, 
            overwrite=True, 
            directory=metadata.dir_path)

    def _set_layers(self, metadata:MetaData)->None:
        self.name = "std_autokeras"
        self.model = ak.StructuredDataRegressor(
            max_trials = metadata.max_trials_autokeras, 
            overwrite=True, 
            directory=metadata.dir_path)


################################################################################
# Std Keras Model Handler
################################################################################
class StdKerasModelHandler(AiModelHandler):

    def _define_model(self, metadata:MetaData)-> None:
        gen_cls=get_from_dict(
            metadata.model_type, KERAS_MODELS, ListKerasModels)
        gen=gen_cls(metadata)
        self.model= gen.get_model()
        self.name = gen.get_name()

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

    def train(self, dataset:AiDatasetHandler, metadata:MetaData)-> None:
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
            steps= None
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
class StdAutokerasModelHandler(AiModelHandler):
    def _define_model(self, metadata:MetaData)-> None:
        gen_cls = get_from_dict(
            metadata.model_type, KERAS_MODELS, ListKerasModels
        )
        gen=gen_cls(metadata)
        self.model=gen.get_model()
        self.name =gen.get_name()
    def _get_specific_var(self, metadata:MetaData)-> None:
        """"""
    def _prepare_model(self)-> None:
        # assert_keras_model_defined(self.model)
        """ """
    def train(self, dataset:AiDatasetHandler, metadata:MetaData)-> None: 
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
            logger.debug(f'{X_pred=}, {X_pred.shape=}')
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
        metadata.optimizer=list(KERAS_OPTIMIZERS.keys())[0].value

    op_cls=get_from_dict(
        metadata.optimizer, KERAS_OPTIMIZERS, ListKerasOptimizers)
    optimizer=op_cls()

    if metadata.learning_rate:
        if metadata.learning_rate >= 1.0:
            raise WrongLearnRateError(f'Wrong learning rate type (>= 1.0): {metadata.learning_rate}') 
        optimizer.learning_rate= metadata.learning_rate

    return optimizer

def get_keras_loss(metadata:MetaData)-> keras.losses.Loss:
    if not metadata.loss:
        metadata.loss=list(KERAS_LOSSES.keys())[0].value

    loss_cls=get_from_dict(metadata.loss, KERAS_LOSSES, ListKerasLosses)
    return loss_cls()

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
""" Dictionary listing all Keras models available
"""
KERAS_MODEL_HANDLERS={
    ListKerasModelHandlers.KerasModelHandler: StdKerasModelHandler,
    ListKerasModelHandlers.AutokerasModelHandler: StdAutokerasModelHandler
}

KERAS_MODELS={
    ListKerasModels.StdKerasModel: StdKerasModel,
    ListKerasModels.StdAutokerasModel: StdAutokerasModel,
}

if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)

    print(ListKerasOptimizers.__name__)

    obj, item=get_from_dict(ListKerasLosses.CategoricalCrossentropy, KERAS_LOSSES, ListKerasLosses, True)
    print(obj(), item)
    obj, item=get_from_dict('CategoricalCrossentropy', KERAS_LOSSES, ListKerasLosses, True)
    print(obj(), item)

    
    """"""
    

    
