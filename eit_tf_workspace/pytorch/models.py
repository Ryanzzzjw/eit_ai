

from typing import Any
from eit_tf_workspace.train_utils.dataset import Datasets
from eit_tf_workspace.train_utils.models import Models, ListModels
from eit_tf_workspace.train_utils.metadata import MetaData

from enum import Enum


from logging import getLogger
logger = getLogger(__name__)

KERAS_MODEL_SAVE_FOLDERNAME='keras_model'

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
    CategoricalCrossentropy='CategoricalCrossentropy'

PYTORCH_LOSS={
    PytorchLosses.CategoricalCrossentropy:''
}

################################################################################
# Std PyTorch Model
################################################################################
class StdPyTorchModel(Models):

    def _define_model(self, metadata:MetaData):
        self.name = "std_pytorch"
        in_size=metadata.input_size
        out_size=metadata.output_size
        # self.model = keras.models.Sequential()
        # self.model.add(keras.layers.Dense(in_size, input_dim = in_size))
        # self.model.add(keras.layers.Activation(tf.nn.relu))
        # self.model.add(keras.layers.Dense(512))
        # self.model.add(keras.layers.Activation(tf.nn.relu))
        # self.model.add(keras.layers.Dense(512))
        # self.model.add(keras.layers.Activation(tf.nn.relu))
        # self.model.add(keras.layers.Dense(out_size)) 
        # self.model.add(keras.layers.Activation(tf.nn.sigmoid))

    def _prepare_model(self, metadata:MetaData):
        if not self.model:
            return
        # optimizer, loss = get_keras_optimizer_loss(metadata)
        # if metadata.learning_rate:
        #     optimizer.learning_rate= metadata.learning_rate 
        # if type(metadata.metrics) != type(list()):
        #     error('metrics need to be a list')
        # self.model.compile( 
        #     optimizer=optimizer,
        #     loss=loss,
        #     metrics=metadata.metrics)

    def train(self, dataset:Datasets, metadata:MetaData):  
        """""" 
        # self.model.fit(
        #     x=dataset.get_X('train'),
        #     y=dataset.get_Y('train'),
        #     epochs=metadata.epoch,
        #     validation_data=(dataset.get_X('val'), dataset.get_Y('val')),
        #     steps_per_epoch=metadata._steps_per_epoch,
        #     validation_steps=metadata._validation_steps,
        #     callbacks=metadata.callbacks,
        #     batch_size=metadata.batch_size)

    def predict(self, dataset:Datasets, metadata:MetaData, **kwargs)->Any:
        """"""
        # return self.model.predict(dataset.get_X('test'), steps=metadata._test_steps)

    def save(self, metadata:MetaData)-> str:
        """"""
        # return save_keras_model(self.model, dir_path=metadata.ouput_dir, save_summary=metadata.save_summary)

    def load(self, metadata:MetaData)-> None:
        """"""
        # self.model=load_keras_model(dir_path=metadata.ouput_dir)


################################################################################
# common methods
################################################################################

# def get_keras_optimizer_loss(metadata:MetaData):

#     if metadata.optimizer not in KERAS_OPTIMIZER.keys():
#         metadata.optimizer=list(KERAS_OPTIMIZER.keys())[0]

#     if metadata.loss not in KERAS_LOSS.keys():
#         metadata.loss=list(KERAS_LOSS.keys())[0]

#     return KERAS_OPTIMIZER[metadata.optimizer](), KERAS_LOSS[metadata.loss]()

# def save_keras_model(model:keras.Model, dir_path:str='', save_summary:bool=False)-> str:
#     """Save a Keras model, additionnaly can be the summary of the model be saved"""
#     if not isdir(dir_path):
#         dir_path=os.getcwd()
#     model_path=os.path.join(dir_path, KERAS_MODEL_SAVE_FOLDERNAME)
#     model.save(model_path)

#     logger.info(f'Keras model saved in: {model_path}')
#     if not save_summary:
#         return
#     summary_path= os.path.join(dir_path, const.MODEL_SUMMARY_FILENAME)
#     with open(summary_path, 'w') as f:
#         with redirect_stdout(f):
#             model.summary()
#     logger.info(f'Keras model summary saved in: {summary_path}')
    
#     return model_path

# def load_keras_model(dir_path:str='') -> keras.models.Model:
#     """Load keras Model and return it if succesful if not """

#     if not isdir(dir_path):
#         logger.info(f'Keras model loading - failed, wrong dir {dir_path}')
#         return
#     model_path=os.path.join(dir_path, KERAS_MODEL_SAVE_FOLDERNAME)
#     if not isdir(model_path):
#         logger.info(f'Keras model loading - failed, {KERAS_MODEL_SAVE_FOLDERNAME} do not exist in {dir_path}')
#         return None
#     try:
#         model:keras.models.Model = keras.models.load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
#         logger.info(f'Keras model loaded: {model_path}')
#         logger.info('Keras model summary:')
#         model.summary()
#         return model
#     except BaseException as e: 
#         logger.error(f'Loading of model from dir: {model_path} - Failed'\
#                      f'\n({e})')
#         return None

################################################################################
# Keras Models
################################################################################

class PyTorchModels(ListModels):
    StdPyTorchModel='StdPyTorchModel'

PYTORCH_MODELS={
    PyTorchModels.StdPyTorchModel: StdPyTorchModel,
}


if __name__ == "__main__":
    from glob_utils.log.log  import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    

    
