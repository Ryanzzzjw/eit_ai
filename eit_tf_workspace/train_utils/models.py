
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from eit_tf_workspace.train_utils.metadata import MetaData
from eit_tf_workspace.train_utils.dataset import Datasets
from enum import Enum

from logging import getLogger
logger = getLogger(__name__)



class WrongOptimizerError(Exception):
    """"""
class WrongLossError(Exception):
    """"""
class WrongMetricsError(Exception):
    """"""
class WrongLearnRateError(Exception):
    """"""
class ModelNotDefinedError(Exception):
    """"""
class ModelNotPreparedError(Exception):
    """"""
################################################################################
# Abstract Class for Models
################################################################################

class ModelManagers(ABC):
    model=None
    name:str=''
    specific_var:dict={}
    
    def __init__(self, metadata:MetaData=None, build:bool=False) -> None:
        super().__init__()
        if build:
            self.build(metadata)
        
    def get_model(self):
        """Return the model object"""
        return self.model

    def get_name(self):
        """Return the name of the model"""
        return self.name

    def build(self, metadata:MetaData=None)->None:
        """Build the model, ready to train"""
        self._define_model(metadata=metadata)
        self._get_specific_var(metadata=metadata)
        self._prepare_model()

    @abstractmethod
    def _define_model(self, metadata:MetaData)-> None:
        """Called first during building:
        Definition of the model layer structure
        >> set self.model"""

    @abstractmethod
    def _get_specific_var(self, metadata:MetaData)-> None:
        """Called second during building:
        Responsible of gathering, preparing, and setting the specific
        variables needed to prepare the model (eg. set optimizer, )
        those variables are stored in the dict specific_var:
        eg.
        specific_var['optimizer']
        specific_var['loss']
        specific_var['metrix']

        Args:
            metadata (MetaData): [description]

        Raises:
            WrongLossError: [description]
            WrongOptimizerError: [description]
            WrongMetrixError: [description]
            WrongLearnRateError: [description]
        """        

    @abstractmethod
    def _prepare_model(self)-> None:
        """ Called thrid during building:
        set the model ready to train (in keras this step is compiling)
        using specific_var
        
        raise  ModelNotDefinedError if model has not been set in self._define_model
        """

    @abstractmethod
    def train(self, dataset:Datasets, metadata:MetaData)-> None:
        """[summary]

        Args:
            dataset (Datasets): [description]
            metadata (MetaData): [description]

        Raises:
            ModelNotDefinedError: if model has not been set in self._define_model
        """        


    @abstractmethod
    def predict(self, dataset:Datasets, metadata:MetaData,**kwargs)->np.ndarray:
        """ """

    @abstractmethod
    def save(self, metadata:MetaData)-> str:
        """Save the current model object

        Args:
            metadata (MetaData): 

        Returns:
            str: the saving path which is automatically set using
            metadata.outputdir
        """        


    @abstractmethod
    def load(self, metadata:MetaData)-> None:
        """ Load a model from the directory metadata.outputdir """


# class ListModels(Enum):
#     """"""

if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    

    
