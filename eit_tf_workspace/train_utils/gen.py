
from abc import ABC, abstractmethod
from datetime import timedelta
from logging import getLogger
import time
from typing import Union
import numpy as np

from eit_tf_workspace.train_utils.models import ModelManagers
from eit_tf_workspace.train_utils.dataset import Datasets
from eit_tf_workspace.train_utils.metadata import MetaData
from eit_tf_workspace.train_utils.lists import ListModels, ListDatasets
from eit_tf_workspace.raw_data.raw_samples import RawSamples

logger = getLogger(__name__)


class WrongModelError(Exception):
    """"""
class WrongDatasetError(Exception):
    """"""

################################################################################
# Abstract Class for ModelGenerator
################################################################################

class Generators(ABC):
    """ Generator abstract class use to manage model and dataset"""
    model_manager:ModelManagers = None
    dataset:Datasets= None

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def select_model_dataset(self, model_type:ListModels=None, dataset_type:ListDatasets=None, metadata:MetaData=None):
        """ """
    @abstractmethod
    def build_dataset(self, raw_samples:RawSamples, metadata:MetaData)-> None:
        """ """
    def extract_samples(self, dataset_part='test', idx_samples:Union[int, list[int], str]=None)-> tuple[np.ndarray,np.ndarray]:
        samples_x, samples_y= self.dataset.get_samples(part=dataset_part)
        if not idx_samples:
                idx_samples= np.random.randint(len(samples_x))
        if isinstance(idx_samples, str):
            if idx_samples.lower()=='all':
                return samples_x, samples_y

        if isinstance(idx_samples, int):
            idx_samples= [idx_samples]

        if isinstance(idx_samples, list):
            samples_x= samples_x[idx_samples]  
            samples_y= samples_y[idx_samples]  
        return samples_x, samples_y
        
    def getattr_dataset(self, attr:str=None):
        return getattr(self.dataset, attr)

    @abstractmethod
    def build_model(self, metadata:MetaData)-> None:
        """ """
    @abstractmethod
    def run_training(self,metadata:MetaData)-> None:
        """ """
    @abstractmethod
    def get_prediction(self,metadata:MetaData, **kwargs)-> np.ndarray:
        """ """
    @abstractmethod
    def save_model(self, metadata:MetaData)-> None:
        """ """
    @abstractmethod
    def load_model(self, metadata:MetaData)-> None:
        """ """

def meas_duration(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result=func(*args, **kwargs)
        duration = timedelta(seconds=time.time() - start_time)
        duration= str(duration)
        if 'return_duration' in kwargs and kwargs.pop('return_duration'):
            return result, duration
        return result
    return wrapper

if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    """"""
