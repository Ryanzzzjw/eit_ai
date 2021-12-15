import random
from logging import getLogger
from typing import Union

import numpy as np
import sklearn.model_selection
from eit_ai.train_utils.dataset import (AiDataset, StdAiDatasetHandler, convert_vec_to_int,
                                        scale_prepocess)
from eit_ai.train_utils.lists import ListPytorchDatasetHandlers
from eit_ai.train_utils.metadata import MetaData
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader


logger = getLogger(__name__)

class StdPytorchDatasetHandler(StdAiDatasetHandler):

    def _post_init(self):
        self.dataset_cls=PytorchDataset

class PytorchDataset(torch.utils.data.Dataset, AiDataset):
    """@Jiawei Please document...

    """    

    def __init__(self, x:np.ndarray, y:np.ndarray)-> None:
        """@Jiawei Please document...

        Args:
            x (np.ndarray): [description]
            y (np.ndarray): [description]

        Raises:
            TypeError: [description]
        """        
        
        if x.shape[0]!=y.shape[0]:
            raise TypeError(
                f'shape not consistent {x.shape[0]}!={y.shape[0]=}, {x=}, {y=}')
            
        self.X = x
        self.Y = y

    def __len__(self):
        """@Jiawei Please document...
        Returns:
            [type]: [description]
        """        
        return len(self.X)

    def __getitem__(
        self,
        idx:Union[int, list[int]]=None)->tuple[torch.Tensor,torch.Tensor]:
        """@Jiawei Please document...

        Args:
            idx (Union[int, list[int]], optional): [description]. Defaults to None.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: [description]
        """        
        return torch.Tensor(self.X[idx]).float(), torch.Tensor(self.Y[idx]).float()
        
    def get_set(self)->tuple[np.ndarray,np.ndarray]:
        """@Jiawei Please document...

        Returns:
            tuple[np.ndarray,np.ndarray]: [description]
        """        
        return self.X, self.Y

class DataloaderGenerator(object):
    def make(
        self,
        dataset:StdPytorchDatasetHandler,
        part:str,
        metadata:MetaData)-> DataLoader:
        """@Jiawei Please document...

        Args:
            dataset (StdPytorchDataset): [description]
            part (str): [description]
            metadata (MetaData): [description]

        Returns:
            torch.utils.data.DataLoader: [description]
        """        
        return DataLoader(getattr(dataset,part), batch_size=metadata.batch_size, shuffle=True, num_workers=0)


PYTORCH_DATASET_HANDLERS={
    ListPytorchDatasetHandlers.StdPytorchDatasetHandler: StdPytorchDatasetHandler
}



if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)

    # X = np.array([[random.randint(0, 100) for _ in range(4)] for _ in range(100)])
    # Y = np.array([random.randint(0, 100) for _ in range(100)])
    # print(f'{X}; {X.shape}\n; {Y}; {Y.shape}')
    
    X = np.random.randn(100, 4)
    Y = np.random.randn(100)
    Y = Y[:, np.newaxis]
    
    rdn_dataset = PytorchDataset(X, Y)
    
    datatset = StdPytorchDatasetHandler()




        

        