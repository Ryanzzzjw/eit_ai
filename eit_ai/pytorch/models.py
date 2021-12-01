import os
# import torch.nn.functional as f
from enum import Enum
from logging import getLogger
from typing import Any

import numpy as np
import torch
from eit_ai.pytorch.const import PYTORCH_LOSS, PYTORCH_OPTIMIZER, PytorchLosses, PytorchOptimizers
from eit_ai.pytorch.dataset import X, DataloaderGenerator, DataloaderGenerator_old, StdPytorchDataset
from eit_ai.train_utils.dataset import Datasets
from eit_ai.train_utils.lists import PytorchModels
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.models import (ListModels, ModelNotDefinedError,
                                       ModelNotPreparedError, Models,
                                       WrongLearnRateError, WrongLossError,
                                       WrongMetricsError, WrongOptimizerError)
from genericpath import isdir
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader

logger = getLogger(__name__)



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
        
    def forward(self, dataloader):
        
        # self.specific_var['optimizer']= get_pytorch_optimizer(metadata)
        # self.specific_var['loss'] = get_pytorch_loss(metadata)
        # loss= nn.MSELoss()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
        # batch = [b.cuda() for b in batch] # if needs GPU
        inputs,  labels = dataloader
        x = self.relu(self.linear1(inputs))
        out = self.relu(self.linear2(x))
        
        self.loss = self.loss(out, labels)
        self.optimizer.zero_grad() 
        self.loss.backward() 
        self.optimizer.step() 
    

    def predict(self, x):
        """[summary]
        """        

        inputs = x
        x = self.relu(self.linear1(inputs))
        out = self.relu(self.linear2(x))
        return out
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
            WrongLossError: raised if passed metadata.loss is not in torch_LOSS list
            WrongOptimizerError: raised if passed metadata.optimizer is not in torch_OPTIMIZER list
            WrongMetrixError: raised if passed metadata.metrics is not a list #Could be better tested... TODO
            WrongLearnRateError: raised if passed metadata.learning_rate >= 1.0 
        """     
        self.specific_var['optimizer'] = get_torch_optimizer(metadata)
        # self.specific_var['lr'] = metadata.learning_rate
        self.specific_var['loss'] = get_torch_loss(metadata)
        if not isinstance(metadata.metrics, list):
            raise WrongMetricsError(f'Wrong metrics type: {metadata.metrics}')
        self.specific_var['metrics'] = metadata.metrics   


    def _prepare_model(self)-> None:
        """3. method called during building:
        set the model ready to train (in torch this step is compiling)
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
        gen=DataloaderGenerator_old()
        train_dataloader=gen.make(dataset, 'train', metadata=metadata)

        for e in range(metadata.epoch):
            for data_i in enumerate(train_dataloader):
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
            metadata.dirpath
        """     
        return save_torch_model(self.model, dir_path=metadata.dir_path)



    def load(self, metadata:MetaData)-> None:
        """Load a model from the directory metadata.model_saving_path 

        Args:
            metadata (MetaData)
        """  
        self.model=load_torch_model(dir_path=metadata.dir_path)


################################################################################
# common methods
################################################################################
PYTORCH_MODEL_SAVE_FOLDERNAME= 'model.torch'

def assert_torch_model_defined(model:Any)-> nn.Module:
    """allow to react if model not  defined

    Args:
        model (Any): [description]

    Raises:
        ModelNotDefinedError: [description]

    Returns:
        torch.models.Model: [description]
    """    
    if not isinstance(model, nn.Module):
        raise ModelNotDefinedError(f'Model has not been correctly defined: {model}')
    return model


def get_torch_optimizer(metadata:MetaData, net:nn.Module)-> torch.optim.Optimizer:

    if not metadata.optimizer:
        metadata.optimizer=list(PYTORCH_OPTIMIZER.keys())[0].value
    try:
        optimizer=PYTORCH_OPTIMIZER[PytorchOptimizers(metadata.optimizer)]
    except ValueError:
        raise WrongOptimizerError(f'Wrong optimizer type: {metadata.optimizer}')

    if metadata.learning_rate:
        if metadata.learning_rate >= 1.0:
            raise WrongLearnRateError(f'Wrong learning rate type (>= 1.0): {metadata.learning_rate}')
        return optimizer(net.parameters(), lr= metadata.learning_rate)
    return optimizer(net.parameters())

def get_torch_loss(metadata:MetaData):

    if not metadata.loss:
        metadata.loss=list(PYTORCH_LOSS.keys())[0].value
    try:
        loss=PYTORCH_LOSS[PytorchLosses(metadata.loss)]()
    except ValueError:
        raise WrongLossError(f'Wrong loss type: {metadata.loss}')

    return loss

def save_torch_model(model:nn.Module, dir_path:str='', save_summary:bool=False)-> str:
    """Save a torch model, additionnaly can be the summary of the model be saved"""
    if not isdir(dir_path):
        dir_path=os.getcwd()
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    
    model.save(model_path)

    logger.info(f'torch model saved in: {model_path}')
    
    # if save_summary:
    #     summary_path= os.path.join(dir_path, const.MODEL_SUMMARY_FILENAME)
    #     with open(summary_path, 'w') as f:
    #         with redirect_stdout(f):
    #             model.summary()
    #     logger.info(f'torch model summary saved in: {summary_path}')
    
    return model_path

def load_torch_model(dir_path:str='') -> nn.Module:
    """Load torch Model and return it if succesful if not """

    if not isdir(dir_path):
        logger.info(f'torch model loading - failed, wrong dir {dir_path}')
        return
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    if not isdir(model_path):
        logger.info(f'torch model loading - failed, {PYTORCH_MODEL_SAVE_FOLDERNAME} do not exist in {dir_path}')
        return None
    try:
        model:nn.Module = torch.load(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        logger.info(f'torch model loaded: {model_path}')
        logger.info('torch model summary:')
        model.summary()
        return model
    except BaseException as e: 
        logger.error(f'Loading of model from dir: {model_path} - Failed'\
                     f'\n({e})')
        return None
################################################################################
# pytorch Models
################################################################################


PYTORCH_MODELS={
    PytorchModels.StdTorchModel: StdTorchModel,
}


if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    

    
