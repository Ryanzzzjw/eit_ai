
from abc import ABC, abstractmethod
from datetime import timedelta
import logging
import time
from typing import Any, Union
import numpy as np

from eit_ai.train_utils.models import AiModelHandler
from eit_ai.train_utils.dataset import AiDatasetHandler
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.lists import ListModelHandlers, ListDatasetHandlers, ListModels
from eit_ai.raw_data.raw_samples import RawSamples
from glob_utils.args.kwargs import kwargs_extract

logger = logging.getLogger(__name__)


class WrongModelError(Exception):
    """"""
class WrongDatasetError(Exception):
    """"""
class WrongSingleXError(Exception):
    """"""

################################################################################
# Abstract Class for AiWorkspaces
################################################################################

class AiWorkspace(ABC):
    """Ai Workspaces abstract class in which a model- and dataset-handler
    co-exist 
    
    """
    model_handler:AiModelHandler = None
    dataset_handler:AiDatasetHandler= None

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def select_model_dataset(
        self,
        model_handler:ListModelHandlers=None,
        dataset_handler:ListDatasetHandlers=None,
        model:ListModels=None,
        metadata:MetaData=None)-> None:
        """Set the model_handler, the model and the dataset_handler,
        from Lists defined for each workspaces type (keras, pythorch, ...)
        (test if they are compatible otherwise raise Error)
        if model_handler and dataset_handler and model are None,
        >> the values contained in metadata will be used

        used types are saved in metadata, if all could be selected correctly
        
        Args:
            model_handler (ListModelHandlers, optional): Defaults to None.
            dataset_handler (ListDatasetHandlers, optional): Defaults to None.
            model (ListModels, optional): Default to None
            metadata (MetaData, optional): Defaults to None.

        Raises:
            WrongModelError: 
            WrongDatasetError:
        """

    @abstractmethod
    def build_dataset(self, raw_samples:RawSamples, metadata:MetaData)-> None:
        """Call the "build"-method of the dataset_handler object

        Args:
            raw_samples (RawSamples): 
            metadata (MetaData): 
        """

    def extract_samples(
        self,
        dataset_part:str='test',
        idx_samples:Union[int, list[int], str]=None
        )-> tuple[np.ndarray,np.ndarray]:  # sourcery skip: merge-nested-ifs
        """Return samples from the part (train, val, test) of the dataset
        as a (x,y) matrix pair. Samples are selected using there idx_samples: 
        
        if idx_samples is an int or list[int], 
        >> the idx_samples will be returned as  (X[idx_samples], Y[idx_samples])

        if idx_samples is 'all',
        >>the complete matrix XY from the dataset part

        if idx_samples is None,
        >> a single and random samples will be return


        Args:
            dataset_part (str, optional): Defaults to 'test'.
            idx_samples (Union[int, list[int], str], optional): Defaults to None.

        Returns:
            tuple[np.ndarray,np.ndarray]: samples_x, samples_y
        """        
        samples_x, samples_y= self.dataset_handler.get_samples(part=dataset_part)
        if idx_samples is None:
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
        
    def getattr_dataset(self, attr:str)->Any:
        """Return an attribute of the dataset

        Args:
            attr (str): attribute name.

        Returns:
            Any: attribute value
        """        
        return getattr(self.dataset_handler, attr)

    @abstractmethod
    def build_model(self, metadata:MetaData)-> None:
        """Call the build method of the model_manager object

        Args:
            metadata (MetaData):
        """        

    @abstractmethod
    def run_training(self,metadata:MetaData, dataset:AiDatasetHandler=None)-> None:
        """Start training with the 'train' and 'val' part of the intern dataset
        or with the passed one.

        Args:
            metadata (MetaData): 
            dataset (Datasets, optional): Defaults to None.
        """        
        
    @abstractmethod
    def get_prediction(
        self,
        metadata:MetaData,
        dataset:AiDatasetHandler=None,
        single_X:np.ndarray= None,
        **kwargs)-> np.ndarray:
        """Return prediction from:
        - 'test'-part of the intern "self.dataset" (if dataset and single_X are None)
        - 'test'-part of the passed dataset (if single_X is None)
        - the single_X (eg. measurements) (fist single_X will be formated)

        Args:
            metadata (MetaData)
            dataset (Datasets, optional): type of "self.dataset". Defaults to None.
            single_X (np.ndarray, optional): array-like of shape (1, n_features). Defaults to None.
            **kwargs: tranmitted to the "predict" method of the model....
        Raises:
            WrongDatasetError: raised if passed dataset is not the same type as "self.dataset"

        Returns:
            np.ndarray: array-like of shape (n_samples, :)
                        predicted samples values
        """        

    @abstractmethod
    def save_model(self, metadata:MetaData)-> None:
        """Call the save model method of the model_manager, 
        and save the saving path in the metadata

        Args:
            metadata (MetaData):
        """        
        
    @abstractmethod
    def load_model(self, metadata:MetaData)-> None:
        """Select the model and dataset (dataset need to be build after) using
        loaded metadata and call the load model method of the model_manager
        to load a trained model and ready to predict!

        Args:
            metadata (MetaData): loaded metadata
            (which contain the type of model and dataset)
        """        

def meas_duration(func):
    """Decorator to meas the time of running 'func'
    if the kwargs return_duration=True, the func will return a tuple with its
    return results and the duration as a str which can be used for logging,...

    eg.:
    @ decorator
    def func():
        return 1
    
    example:    result_func, duration = func(return_duration=True)\n
                result_func = func()

    Args:
        func ([type]): [description]
    """    
    def wrapper(*args, **kwargs)-> Union[tuple[Any, str], Any]:
        return_duration= None
        start_time = time.time()
        return_duration=kwargs_extract(kwargs, 'return_duration')
        result=func(*args, **kwargs)
        duration = timedelta(seconds=time.time() - start_time)
        duration= str(duration)
        
        return result, duration if return_duration else result
        
    return wrapper

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    """"""
