

from typing import Any
from eit_tf_workspace.train_utils.dataset import Datasets
from eit_tf_workspace.train_utils.models import ModelManagers, ListModels
from eit_tf_workspace.train_utils.metadata import MetaData
import torch
from torch import nn
from enum import Enum


from logging import getLogger
logger = getLogger(__name__)

PYTORCH_MODEL_SAVE_FOLDERNAME='keras_model'

################################################################################
# Optimizers
################################################################################

class PytorchOptimizers(Enum):
    Adam='Adam'

PYTORCH_OPTIMIZER={
    PytorchOptimizers.Adam:''
}
################################################################################
# Losses
################################################################################

class PytorchLosses(Enum):
    MSELoss='MSELoss'

PYTORCH_LOSS={
    PytorchLosses.MSELoss: nn.MSELoss
}

################################################################################
# Std PyTorch ModelManager
################################################################################
class StdKerasModel(ModelManagers):

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
        if not isinstance(metadata.metrics ,list):
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

    def predict(self, dataset:Datasets, metadata:MetaData, **kwargs)-> np.ndarray:
        assert_keras_model_compiled(self.model)
        return self.model.predict(dataset.get_X('test'), steps=metadata._test_steps)

    def save(self, metadata:MetaData)-> str:
        assert_keras_model_compiled(self.model)
        return save_keras_model(self.model, dir_path=metadata.ouput_dir, save_summary=metadata.save_summary)

    def load(self, metadata:MetaData)-> None:
        self.model=load_keras_model(dir_path=metadata.ouput_dir)
        assert_keras_model_compiled(self.model)






class stdModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 3),
                                        nn.BatchNorm1d(3),
                                        nn.ReLU(),
                                        nn.Linear(3, 1)
        )
            

    def forward(self, x):
        return self.layers(x)



################################################################################
# common methods
################################################################################
PYTORCH_MODEL_SAVE_FOLDERNAME= 'pytorch_model'

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
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    
    model.save(model_path)

    logger.info(f'Keras model saved in: {model_path}')
    
    if save_summary:
        summary_path= os.path.join(dir_path, const.MODEL_SUMMARY_FILENAME)
        with open(summary_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        logger.info(f'Keras model summary saved in: {summary_path}')
    
    return model_path

def load_keras_model(dir_path:str='') -> keras.models.Model:
    """Load keras Model and return it if succesful if not """

    if not isdir(dir_path):
        logger.info(f'Keras model loading - failed, wrong dir {dir_path}')
        return
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    if not isdir(model_path):
        logger.info(f'Keras model loading - failed, {PYTORCH_MODEL_SAVE_FOLDERNAME} do not exist in {dir_path}')
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
################################################################################
# pytorch Models
################################################################################

class PyTorchModels(ListModels):
    StdPyTorchModel='StdPyTorchModel'

PYTORCH_MODELS={
    PyTorchModels.StdPyTorchModel: StdPyTorchModel,
}


if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    

    
