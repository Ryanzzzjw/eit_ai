

from typing import Any

from eit_ai.train_utils.dataset import Datasets
from eit_ai.train_utils.models import Models, ListModels
from eit_ai.train_utils.metadata import MetaData
from torch.utils import data
import torch
from torch import nn
import torch.nn.functional as f
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
class StdTorchModel(nn.Module):
    def __init__(self, metadata: MetaData) -> None:
        super(StdTorchModel, self).__init__()
        in_size=metadata.input_size
        out_size=metadata.output_size
        self.linear1 = nn.Linear(in_size, 3)
        self.linear2 = nn.Linear(3, out_size)
        self.relu = nn.ReLU()

    def prepare(self, op, loss):
        self.optimizer= op
        self.loss= loss
        
    def forward(self, data):
        
        # self.specific_var['optimizer']= get_pytorch_optimizer(metadata)
        # self.specific_var['loss'] = get_pytorch_loss(metadata)
        # loss= nn.MSELoss()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
        # batch = [b.cuda() for b in batch] # if needs GPU
        input,  labels = data
        x = self.relu(self.linear1(input))
        out = self.relu(self.linear2(x))
        
        self.loss = self.loss(out, labels)
        self.optimizer.zero_grad() 
        self.loss.backward() 
        self.optimizer.step() 
        return out

    def predict(self):
        """[summary]
        """        

    # def save(self, metadata:MetaData)-> str:
        
    #     return torch.save(self.model, dir_path=metadata.ouput_dir, save_summary=metadata.save_summary)

    # def load(self, metadata:MetaData)-> None:
        
        
    #     return torch.load(dir_path=metadata.ouput_dir)
        
class StdPytorchModelManager(Models):

    # model=None
    # name:str=''
    # specific_var:dict={}

    def _define_model(self, metadata:MetaData)-> None:
        """1. method called during building:
        here have should be the model layer structure defined and
        stored in set "self.model"

        Args:
            metadata (MetaData):
        """
        self.model= StdTorchModel(metadata=metadata)
        self.name= 'StdTorch'

 
    def _get_specific_var(self, metadata:MetaData)-> None:
        """2. method called during building:
        Responsible of gathering, preparing, and setting the specific
        variables needed to prepare the model (eg. optimizer, )
        those variables are stored in the dict "specific_var", which can be set
        depending the needs of the model, eg. :
            specific_var['optimizer']
            specific_var['loss']
            specific_var['metrix']

        Args:
            metadata (MetaData): 

        Raises:
            WrongLossError: raised if passed metadata.loss is not in KERAS_LOSS list
            WrongOptimizerError: raised if passed metadata.optimizer is not in KERAS_OPTIMIZER list
            WrongMetrixError: raised if passed metadata.metrics is not a list #Could be better tested... TODO
            WrongLearnRateError: raised if passed metadata.learning_rate >= 1.0 
        """     
        self.specific_var['optimizer'] =get_torch_optimizer(metadata)
        self.specific_var['lr'] metadata.learning_rate
        self.specific_var['loss']
        self.specific_var['metrix']   


    def _prepare_model(self)-> None:
        """3. method called during building:
        set the model ready to train (in keras this step is compiling)
        using "specific_var"

        Raises:
            ModelNotDefinedError: if "self.model" is not a Model type
        """   
        self.model.prepare( 
            self.specific_var['optimizer'],
            self.specific_var['loss']  )



    def train(self, dataset:Datasets, metadata:MetaData)-> None:
        """Train the model with "train" and "val"-part of the dataset, with the
        metadata. Before training the model is tested if it exist and ready

        Args:
            dataset (Datasets): 
            metadata (MetaData):

        Raises:
            ModelNotDefinedError: if "self.model" is not a Model type
            ModelNotPreparedError: if "self.model" is not compiled or similar
        """     
        for e in metadata.epoch:
            for data_i in dataset.train:
                self.model.forward(data_i)   


    def predict(
        self,
        X_pred:np.ndarray,
        metadata:MetaData,
        **kwargs)->np.ndarray:
        """return prediction for features X_pred with the metadata
        Before prediction the model is tested if it exist and ready

        Args:
            X_pred (np.ndarray): array-like of shape (n_samples, :)
                                input values
            metadata (MetaData)

        Raises:
            ModelNotDefinedError: if "self.model" is not a Model type
            ModelNotPreparedError: if "self.model" is not compiled or similar

        Returns:
            np.ndarray: array-like of shape (n_samples, :)
                        predicted samples values
        """
        # X_pred preprocess if needed
        return self.model.predict(X_pred, metadata)  


    def save(self, metadata:MetaData)-> str:
        """Save the current model object

        Args:
            metadata (MetaData)

        Returns:
            str: the saving path which is automatically set using
            metadata.outputdir
        """     
        return save_torch_model(self.model, dir_path=metadata.dir_path, save_summary=metadata.save_summary)



    def load(self, metadata:MetaData)-> None:
        """Load a model from the directory metadata.model_saving_path 

        Args:
            metadata (MetaData)
        """  



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


def get_torch_optimizer(metadata:MetaData)-> keras.optimizers.Optimizer:

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

def save_torch_model(model:keras.Model, dir_path:str='', save_summary:bool=False)-> str:
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
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    

    
