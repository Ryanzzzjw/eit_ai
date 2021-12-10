import random
from logging import getLogger
from typing import Union

import numpy as np
import sklearn.model_selection
from eit_ai.train_utils.dataset import (AiDataset, convert_vec_to_int,
                                        scale_prepocess)
from eit_ai.train_utils.lists import ListPytorchDatasets
from eit_ai.train_utils.metadata import MetaData
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader


logger = getLogger(__name__)

class StdPytorchDataset(AiDataset):
   
    def get_X(self, part:str='train'):
        return getattr(self, part).get_set()[0]

    def get_Y(self, part:str='train'):
        return getattr(self, part).get_set()[1]

    def get_samples(self, part:str='train'):
        return getattr(self, part).get_set()

    def _preprocess(
        self,
        X:np.ndarray,
        Y:np.ndarray,
        metadata:MetaData
        )->tuple[Union[np.ndarray,None],Union[np.ndarray,None]]:
        """return X, Y preprocessed"""
        
        X=scale_prepocess(X, metadata.normalize[0])
        Y=scale_prepocess(Y, metadata.normalize[1])
        if Y is not None:
            logger.debug(f'Size of X and Y (after preprocess): {X.shape=}, {Y.shape=}')     
        else:
            logger.debug(f'Size of X (after preprocess): {X.shape=}')
        return X, Y

    def _mk_dataset(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """build the dataset"""
        idx=np.reshape(range(X.shape[0]),(X.shape[0],1))
        X= np.concatenate(( X, idx ), axis=1)
        x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(
            X, Y,test_size=self._test_ratio)
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
            x_tmp, y_tmp, test_size=self._val_ratio)
        
        self._idx_train= x_train[:,-1].tolist()
        self._idx_val= x_val[:,-1].tolist()
        self._idx_test= x_test[:,-1].tolist()
        metadata.set_idx_samples(self._idx_train, self._idx_val, self._idx_test)

        self.train=PytorchDataset(x=x_train[:,:-1], y=y_train)
        self.val=PytorchDataset(x=x_val[:,:-1], y=y_val)
        self.test=PytorchDataset(x=x_test[:,:-1], y=y_test)
        
    def _mk_dataset_from_indexes(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """rebuild the dataset with the indexes """
        self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
        self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
        self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
        self.train=PytorchDataset(x=X[self._idx_train,:], y=Y[self._idx_train,:])
        self.val=PytorchDataset(x=X[self._idx_val,:], y=Y[self._idx_val,:])
        self.test=PytorchDataset(x=X[self._idx_test,:], y=Y[self._idx_test,:])

class PytorchDataset(torch.utils.data.Dataset):
    """@Jiawei Please document...

    """    

    def __init__(self, x:np.ndarray, y:np.ndarray)-> None:
        """@Jiawei Please document...

        Args:
            x (np.ndarray): ArrayLike (n_samples, n_features)
            y (np.ndarray): ArrayLike (n_samples, n_labels)
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

# class DataloaderGenerator(object):
#     def make(self, dataset:PytorchDataset, metadata:MetaData)->DataLoader:
#         # self.train = StdPytorchDataset().train
#         # self.val = StdPytorchDataset().val
#         # self.test = StdPytorchDataset().test
#         return DataLoader(dataset, batch_size=metadata.batch_size, shuffle=True, num_workers=0)
class DataloaderGenerator(object):
    def make(
        self,
        dataset:StdPytorchDataset,
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


PYTORCH_DATASETS={
    ListPytorchDatasets.StdPytorchDataset: StdPytorchDataset
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
    
    datatset = StdPytorchDataset()




        

        