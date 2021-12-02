import os
# import torch.nn.functional as f
from enum import Enum
from logging import getLogger
from typing import Any
from contextlib import redirect_stdout
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

MODEL_SUMMARY_FILENAME='model_summary'

################################################################################
# Std PyTorch ModelManager
################################################################################
class StdPytorchModel(nn.Module):
    def __init__(self, metadata: MetaData) -> None:
        super(StdPytorchModel, self).__init__()
        in_size=metadata.input_size
        out_size=metadata.output_size
        # self.linear1 = nn.Linear(in_size, 3)
        # self.linear2 = nn.Linear(3, out_size)
        # self.relu = nn.ReLU()
        self.model = torch.nn.Sequential()
        self.model.add_module('dense1', nn.Linear(in_size, 512))
        self.model.add_module('relu', nn.ReLU())
        self.model.add_module('dense2', nn.Linear(512, out_size))
        self.model.add_module('sigmoid', nn.Sigmoid())

    def prepare(self, op:torch.optim.Optimizer, loss):
        self.optimizer= op
        self.loss= loss
        
    def forward(self, dataloader):
        
        # inputs,  labels = dataloader
        # x = self.relu(self.linear1(inputs))
        return self.model(dataloader)
        
        # self.loss = self.loss(out, labels)
        # self.optimizer.zero_grad() 
        # self.loss.backward()
        # self.optimizer.step() 
    

    def predict(self, x_pred: np.ndarray):
        """[summary]
        """
        self.model.eval()
        with torch.no_grad:
            out = self.model(torch.Tensor(x_pred))        
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
        self.specific_var['optimizer'] = get_pytorch_optimizer(metadata)
        # self.specific_var['lr'] = metadata.learning_rate
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

        for epoch in range(metadata.epoch):
            logger.info(f'Epoch {epoch}/{metadata.epoch}')
            for idx, data_i in enumerate(train_dataloader):
                logger.debug(f'Batch {idx}')
                inputs, labels = data_i
                y_pred = self.model(inputs)
                loss_value = self.model.loss(y_pred, labels)
               
                self.model.optimizer.zero_grad()
                loss_value.backward()
                self.model.optimizer.step() 
                if (epoch+1) % 20 == 0:
                    print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1,
                                                  metadata.epoch,
                                                  loss_value.item()))
                
                


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
        if X_pred.shape[0]==1:
            return self.model.predict(X_pred)
        else:
            res = []
            for i in range(X_pred.shape[0]):
                pred = self.model.predict(X_pred[[i], :]).numpy().tolist()
                res.append[pred]
                return res  


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
PYTORCH_MODEL_SAVE_FOLDERNAME= 'model.pytorch'

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
    try:
        optimizer=PYTORCH_OPTIMIZER[PytorchOptimizers(metadata.optimizer)]
    except ValueError:
        raise WrongOptimizerError(f'Wrong optimizer type: {metadata.optimizer}')

    if metadata.learning_rate:
        if metadata.learning_rate >= 1.0:
            raise WrongLearnRateError(f'Wrong learning rate type (>= 1.0): {metadata.learning_rate}')
        return optimizer(net.parameters(), lr= metadata.learning_rate)
    return optimizer(net.parameters(), lr=metadata.learning_rate)

def get_pytorch_loss(metadata:MetaData):

    if not metadata.loss:
        metadata.loss=list(PYTORCH_LOSS.keys())[0].value
    try:
        loss=PYTORCH_LOSS[PytorchLosses(metadata.loss)]()
    except ValueError:
        raise WrongLossError(f'Wrong loss type: {metadata.loss}')

    return loss

def save_pytorch_model(model:nn.Module, dir_path:str='', save_summary:bool=False)-> str:
    """Save a pytorch model, additionnaly can be the summary of the model be saved"""
    if not isdir(dir_path):
        dir_path=os.getcwd()
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    
    torch.save(model, model_path)

    logger.info(f'pytorch model saved in: {model_path}')
    
    if save_summary:
        summary_path= os.path.join(dir_path, MODEL_SUMMARY_FILENAME)
        with open(summary_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        logger.info(f'pytorch model summary saved in: {summary_path}')
    
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
        model:nn.Module = torch.load(model_path)
        logger.info(f'pytorch model loaded: {model_path}')
        logger.info('pytorch model summary:')
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
    PytorchModels.StdPytorchModel: StdPytorchModel,
}


if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    

    
