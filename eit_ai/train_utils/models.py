
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.dataset import AiDatasetHandler
from enum import Enum

import logging
logger = logging.getLogger(__name__)

MODEL_SUMMARY_FILENAME='model_summary'

################################################################################
# Custom Exeptions/Errors for Models
################################################################################
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

class AiModelHandler(ABC):

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

    def build(self, metadata:MetaData)->None:
        """Build the model, ready to train
        this pipeline is defined by three internal methods
        
        self._define_model(metadata=metadata)
        self._get_specific_var(metadata=metadata)
        self._prepare_model()

        Args:
            metadata (MetaData):
        """        
        self._define_model(metadata=metadata)
        self._get_specific_var(metadata=metadata)
        self._prepare_model()

    @abstractmethod
    def _define_model(self, metadata:MetaData)-> None:
        """1. method called during building:
        here have should be the model layer structure defined and
        stored in set "self.model"

        Args:
            metadata (MetaData):
        """

    @abstractmethod
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

    @abstractmethod
    def _prepare_model(self)-> None:
        """3. method called during building:
        set the model ready to train (in keras this step is compiling)
        using "specific_var"

        Raises:
            ModelNotDefinedError: if "self.model" is not a Model type
        """        


    @abstractmethod
    def train(self, dataset:AiDatasetHandler, metadata:MetaData)-> None:
        """Train the model with "train" and "val"-part of the dataset, with the
        metadata. Before training the model is tested if it exist and ready

        Args:
            dataset (Datasets): 
            metadata (MetaData):

        Raises:
            ModelNotDefinedError: if "self.model" is not a Model type
            ModelNotPreparedError: if "self.model" is not compiled or similar
        """        

    @abstractmethod
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

    @abstractmethod
    def save(self, metadata:MetaData)-> str:
        """Save the current model object

        Args:
            metadata (MetaData)

        Returns:
            str: the saving path which is automatically set using
            metadata.outputdir
        """        

    @abstractmethod
    def load(self, metadata:MetaData)-> None:
        """Load a model from the directory metadata.model_saving_path 

        Args:
            metadata (MetaData)
        """  


# class ListModels(Enum):
#     """"""

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    

    
