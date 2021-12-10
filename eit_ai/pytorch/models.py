from abc import ABC, abstractmethod
import os
# import torch.nn.functional as f
from enum import Enum
from logging import getLogger
from typing import Any
from contextlib import redirect_stdout
import numpy as np
import torch
from eit_ai.pytorch.const import PYTORCH_LOSS, PYTORCH_MODEL_SAVE_FOLDERNAME, PYTORCH_OPTIMIZER, PytorchLosses, PytorchOptimizers
from eit_ai.pytorch.dataset import DataloaderGenerator, StdPytorchDataset, PytorchDataset

from eit_ai.train_utils.dataset import Datasets
from eit_ai.train_utils.lists import ListPyTorchLosses, ListPyTorchOptimizers, ListPytorchModels, PytorchModels, get_from_dict
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.models import (MODEL_SUMMARY_FILENAME, ModelNotDefinedError,
                                       ModelNotPreparedError, Models,
                                       WrongLearnRateError, WrongLossError,
                                       WrongMetricsError, WrongOptimizerError)
from genericpath import isdir
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader

logger = getLogger(__name__)

class StdPytorch(ABC):
    model:nn.Module=None
    def __init__(self, metadata: MetaData) -> None:
        super().__init__()
        self._set_layers(metadata)
    
    @abstractmethod
    def _set_layers(self, metadata:MetaData)-> None:
        """[summary]

        Args:
            metadata (MetaData): [description]
        """ 
    
    def prepare(self, op:torch.optim.Optimizer, loss):
        self.optimizer= op
        self.loss= loss
        
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        logger.debug(f'foward, {x.shape=}')
        return self.model(x)

    def run_single_epoch(self, dataloader:DataLoader)->Any:
        logger.debug(f'run_single_epoch')
        for idx, data_i in enumerate(dataloader):
            logger.debug(f'Batch #{idx}')
            inputs, labels = data_i
            y_pred = self.forward(inputs)
            #loss
            loss_value = self.loss(y_pred, labels)
            
            #backward propagation
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()  #update
            logger.debug(f'Batch #{idx}: loss={loss_value.item():.6f}')
        return loss_value.item() 

    def get_net(self):
        return self.model

    def predict(self, x_pred: np.ndarray)->np.ndarray:
        """[summary]
        predict the new x
        """
        return self.forward(torch.Tensor(x_pred)).detach().numpy()

class StdPytorchModel(StdPytorch):

    def _set_layers(self, metadata:MetaData)-> None:
        in_size=metadata.input_size
        out_size=metadata.output_size
        self.model = torch.nn.Sequential()
        self.model.add_module('dense1', nn.Linear(in_size, 512))
        self.model.add_module('relu', nn.ReLU())
        self.model.add_module('dense2', nn.Linear(512, out_size))
        self.model.add_module('sigmoid', nn.Sigmoid())
    
################################################################################
# Std PyTorch ModelManager
################################################################################
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
        self.model= StdPytorchModel(metadata=metadata)
        self.name= 'StdPytorch'

 
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
        self.specific_var['optimizer'] = get_pytorch_optimizer(metadata, self.model.get_net())
        self.specific_var['loss'] = get_pytorch_loss(metadata)
        if not isinstance(metadata.metrics, list):
            raise WrongMetricsError(f'Wrong metrics type: {metadata.metrics}')
        self.specific_var['metrics'] = metadata.metrics   


    def _prepare_model(self)-> None:
        """3. method called during building:
        set the model ready to train (in pytorch this step is compiling)
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
        gen=DataloaderGenerator()
        train_dataloader=gen.make(dataset, 'train', metadata=metadata)
        logger.info(f'Training - Started {metadata.epoch}')
        for epoch in range(metadata.epoch):
            loss= self.model.run_single_epoch(train_dataloader)
            logger.info(f'Epoch #{epoch+1}/{metadata.epoch} : {loss=}')   


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
        # if X_pred.shape[0]==1:
        #     return self.model.predict(X_pred)
        # else:
        #     res = np.array([])
        #     for i in range(X_pred.shape[0]):
        #         pred = self.model.predict(X_pred[i])
        #         res = np.append(res, pred)
        #     return res  

        return self.model.predict(X_pred)

    def save(self, metadata:MetaData)-> str:
        """Save the current model object

        Args:
            metadata (MetaData)

        Returns:
            str: the saving path which is automatically set using
            metadata.dirpath
        """     
        return save_pytorch_model(self.model, dir_path=metadata.dir_path, save_summary=metadata.save_summary)



    def load(self, metadata:MetaData)-> None:
        """Load a model from the directory metadata.model_saving_path 

        Args:
            metadata (MetaData)
        """  
        self.model= load_pytorch_model(dir_path=metadata.dir_path)


################################################################################
# common methods
################################################################################

def assert_pytorch_model_defined(model:Any)-> nn.Module:
    """allow to react if model not  defined

    Args:
        model (Any): [description]

    Raises:
        ModelNotDefinedError: [description]

    Returns:
        pytorch.models.Model: [description]
    """    
    if not isinstance(model, nn.Module):
        raise ModelNotDefinedError(f'Model has not been correctly defined: {model}')
    return model


def get_pytorch_optimizer(metadata:MetaData, net:nn.Module)-> torch.optim.Optimizer:

    if not metadata.optimizer:
        metadata.optimizer=list(PYTORCH_OPTIMIZER.keys())[0].value

    op_cls=get_from_dict(
        metadata.optimizer, PYTORCH_OPTIMIZER, ListPyTorchOptimizers)
    optimizer=op_cls()

    if metadata.learning_rate:
        if metadata.learning_rate >= 1.0:
            raise WrongLearnRateError(f'Wrong learning rate type (>= 1.0): {metadata.learning_rate}') 
        return optimizer(net.parameters(), lr= metadata.learning_rate)
    
    logger.warning('Learningrate has been set to 0.001!!!')
    return optimizer(net.parameters(), lr=0.001)
        

def get_pytorch_loss(metadata:MetaData)->nn.modules.loss:

    if not metadata.loss:
        metadata.loss=list(PYTORCH_LOSS.keys())[0].value

    loss_cls=get_from_dict(metadata.loss, PYTORCH_LOSS, ListPyTorchLosses)
    return loss_cls()

def save_pytorch_model(model:nn.Module, dir_path:str='', save_summary:bool=False)-> str:
    """Save a pytorch model, additionnaly can be the summary of the model be saved"""
    if not isdir(dir_path):
        dir_path=os.getcwd()
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    
    torch.save(model, model_path)

    logger.info(f'PyTorch model saved in: {model_path}')
    
    if save_summary:
        logger.info('pytorch summary saving is not implemented')

        # from torchvision import models
        # from torchsummary import summary

        # vgg = models.vgg16()
        # summary(vgg, (3, 224, 224))

        # summary_path= os.path.join(dir_path, MODEL_SUMMARY_FILENAME)
        # with open(summary_path, 'w') as f:
        #     with redirect_stdout(f):
        #         model.summary()
        # logger.info(f'pytorch model summary saved in: {summary_path}')
    
    return model_path

def load_pytorch_model(dir_path:str='') -> nn.Module:
    """Load pytorch Model and return it if succesful if not """

    if not isdir(dir_path):
        logger.info(f'pytorch model loading - failed, wrong dir {dir_path}')
        return
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    if not isdir(model_path):
        logger.info(f'pytorch model loading - failed, {PYTORCH_MODEL_SAVE_FOLDERNAME} do not exist in {dir_path}')
        return None
    try:
        model= torch.load(model_path)
        logger.info(f'pytorch model loaded: {model_path}')
        logger.info('pytorch model summary:')
        # model.summary() NotImplemented yet
        return model
    except BaseException as e: 
        logger.error(f'Loading of model from dir: {model_path} - Failed'\
                     f'\n({e})')
        return None
################################################################################
# pytorch Models
################################################################################


PYTORCH_MODELS={
    ListPytorchModels.StdPytorchModelManager: StdPytorchModelManager,
}


if __name__ == "__main__":
    import logging
    from eit_ai.raw_data.matlab import MatlabSamples
    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    


    X = np.random.randn(100, 4)
    Y = np.random.randn(100)
    Y = Y[:, np.newaxis]

    raw=MatlabSamples()
    raw.X=X
    raw.Y=Y
    
    # rdn_dataset = PytorchDataset(X, Y)
    md= MetaData()
    md.set_4_dataset()
    dataset = StdPytorchDataset()
    dataset.build(raw_samples=raw, metadata= md)

    new_model = StdPytorchModelManager()
    # for epoch in range(50):
    md.set_4_model()
    new_model.train(dataset,md)
