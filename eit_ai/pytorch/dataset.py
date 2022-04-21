from operator import mod
import random
import logging
from typing import Tuple, Union

import numpy as np
import sklearn.model_selection
from eit_ai.train_utils.dataset import (AiDataset, StdAiDatasetHandler, convert_vec_to_int,
                                        scale_preprocess)
from eit_ai.train_utils.lists import ListPytorchDatasetHandlers, ListPytorchModelHandlers, ListPytorchModels
from eit_ai.train_utils.metadata import MetaData
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)

class StdPytorchDatasetHandler(StdAiDatasetHandler):

    def _post_init(self):
        self.dataset_cls=PytorchDataset
        

class PytorchDataset(torch.utils.data.Dataset, AiDataset):
    """ create the customized Pytorch dataset """    

    def __init__(self, x:np.ndarray, y:np.ndarray)-> None:
        """ load the original X and Y.

        Args:
            x (np.ndarray): [description]
            y (np.ndarray): [description]

        Raises:
            TypeError: [description]
        """        
        
        if x.shape[0]!=y.shape[0]:
            raise TypeError(
                f'shape not consistent {x.shape[0]}!={y.shape[0]=}, {x=}, {y=}')
            
        self.X = -x
        self.Y = y

    def __len__(self):
        """ return the number of samples.
        Returns:
            [type]: [description]
        """        
        return len(self.X)

    def __getitem__(
        self,
        idx:Union[int, list[int]]=None)->tuple[torch.Tensor,torch.Tensor]:
        """convert array to tensor. And allow to return a sample with the given index.

        Args:
            idx (Union[int, list[int]], optional): [description]. Defaults to None.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: [description]
        """        
        x,y= torch.Tensor(self.X[idx]).float(), torch.Tensor(self.Y[idx]).float()
        return x, y
        
    def get_set(self)->tuple[np.ndarray,np.ndarray]:
        """ return X and Y separately.

        Returns:
            tuple[np.ndarray,np.ndarray]: [description]
        """        
        return self.X, self.Y

class PytorchConv1dDatasetHandler(StdAiDatasetHandler):

    def _post_init(self):
        self.dataset_cls=PytorchConv1dDataset


class PytorchConv1dDataset(torch.utils.data.Dataset, AiDataset):
    """ create the customized Pytorch dataset """    

    def __init__(self, x:np.ndarray, y:np.ndarray)-> None:
        """ load the original X and Y.

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
        self.X_conv= reshape_4_1Dconv(-x) # special reshape for conv Ai
        self.Y = y

    def __len__(self):
        """ return the number of samples.
        Returns:
            [type]: [description]
        """        
        return len(self.X)

    def __getitem__(
        self,
        idx:Union[int, list[int]]=None)->tuple[torch.Tensor,torch.Tensor]:
        """convert array to tensor. And allow to return a sample with the given index.

        Args:
            idx (Union[int, list[int]], optional): [description]. Defaults to None.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: [description]
        """        
        x,y= torch.Tensor(self.X_conv[idx]).float(), torch.Tensor(self.Y[idx]).float()
        return x, y
        
    def get_set(self)->tuple[np.ndarray,np.ndarray]:
        """ return X and Y separately.

        Returns:
            tuple[np.ndarray,np.ndarray]: [description]
        """        
        return self.X, self.Y

class PytorchUxyzDatasetHandler(StdAiDatasetHandler):

    def _post_init(self):
        self.dataset_cls=PytorchUxyzDataset
    
    def _mk_dataset(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """build the dataset"""
        idx=np.reshape(range(X.shape[0]),(X.shape[0],1))
        X= np.concatenate(( X, idx ), axis=1)
        x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y,test_size=self._test_ratio)
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp, test_size=self._val_ratio)
        
        self._idx_train= x_train[:,-1].tolist()
        self._idx_val= x_val[:,-1].tolist()
        self._idx_test= x_test[:,-1].tolist()
        metadata.set_idx_samples(self._idx_train, self._idx_val, self._idx_test)

        self.train=self.dataset_cls(x_train[:,:-1], y_train, self.fwd_model)
        self.val=self.dataset_cls(x_val[:,:-1], y_val, self.fwd_model)
        self.test=self.dataset_cls(x_test[:,:-1], y_test, self.fwd_model)

    def _mk_dataset_from_indexes(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """rebuild the dataset with the indexes """
        self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
        self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
        self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
        self.train=self.dataset_cls(X[self._idx_train,:], Y[self._idx_train,:],self.fwd_model)
        self.val=self.dataset_cls(X[self._idx_val,:], Y[self._idx_val,:], self.fwd_model)
        self.test=self.dataset_cls(X[self._idx_test,:], Y[self._idx_test,:],self.fwd_model)    

class PytorchUxyzDataset(torch.utils.data.Dataset, AiDataset):
    """ create the customized Pytorch dataset """    

    def __init__(self, x:np.ndarray, y:np.ndarray, fwd_model)-> None:
        """ load the original X and Y.

        Args:
            x (np.ndarray): [description]
            y (np.ndarray): [description]

        Raises:
            TypeError: [description]
        """        
        
        if x.shape[0]!=y.shape[0]:
            raise TypeError(
                f'shape not consistent {x.shape[0]}!={y.shape[0]=}, {x=}, {y=}')
            
        self.X = x # U (N, n_meas)
        self.Y = y # sigma elem_data (N, n_elem)

        # self._new_X= np.array([]) # [U, x,y,z] (N*n_pos, n_meas+3)
        # self._new_Y= np.array([])# conductitiy for n postions (N*n_pos, 1)

        logger.debug("Generation of positions - Start ...")
        self.pts = fwd_model['nodes']
        self.tri = fwd_model['elems']
        self.center_e = np.mean(self.pts[self.tri], axis=1)
        self.n_pos=len(self.center_e)
          
        #TODO Generation of pos and c
        # self.pos= np.array([]) # (n_pos, 3, N)
        # self.cpos= np.array([]) # (n_pos, 1, N)
        logger.debug("Generation of positions - Done")  

    def __len__(self):
        """ return the number of samples.
        Returns:
            [type]: [description]
        """        
        return len(self.X)*self.n_pos

    def __getitem__(
        self,
        idx:Union[int, list[int]]=None)->tuple[torch.Tensor,torch.Tensor]:
        """convert array to tensor. And allow to return a sample with the given index.

        Args:
            idx (Union[int, list[int]], optional): [description]. Defaults to None.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: [description]
        """        
        # x,y= torch.Tensor(self.X_conv[idx]).float(), torch.Tensor(self.Y[idx]).float()
        new_X = np.empty((0,259))
        new_Y = np.empty((0,1))
        for i in range(idx+1):
            temp_x, temp_y = self.build_Uxyz_c(i)
            np.vstack((new_X, temp_x))
            np.vstack((new_Y, temp_y))
        logger.debug(f'newY ={new_X}, newY shape={new_X.shape}')
        logger.debug(f'newY ={new_Y}, newY shape={new_Y.shape}')
        
        return torch.Tensor(new_X).float(), torch.Tensor(new_Y).float()
        
    def get_set(self)->tuple[np.ndarray,np.ndarray]:
        """ return X and Y separately.

        Returns:
            tuple[np.ndarray,np.ndarray]: [description]
        """        
        return self.X, self.Y

    def build_Uxyz_c(self, idx:Union[int, list[int]])-> Tuple[torch.Tensor, torch.Tensor]:
        self.pos_batch, self.cpos_batch= self.get_pos_c_batch(idx)
        
        # here use idx, self.X , self.Y, self.pos_batch, self.cpos_batch
        # to build  self._new_X , self._new_Y
        m_pos = mod(idx, self.n_pos)
        n_samples = idx // self.n_pos
        self._new_Y=self.Y[n_samples, m_pos].reshape(1, -1)
        self._new_X=np.hstack((self.X[n_samples,:], self.center_e[m_pos,:])).reshape(1, -1)
        logger.debug(f'newX ={self._new_X}, newY shape={self._new_X.shape}')
        logger.debug(f'newY ={self._new_Y}, newY shape={self._new_Y.shape}')
        return self._new_X , self._new_Y# conductitiy for n postions (N*n_pos, 1)

    # def get_pos_c_batch(self, idx:Union[int, list[int]])-> None:
        # here set self.pos_batch, self.cpos_batch
        pos_batch, cpos_batch= None, None
        return pos_batch, cpos_batch


class DataloaderGenerator(object):
    def make(
        self,
        dataset:StdPytorchDatasetHandler,
        part:str,
        metadata:MetaData)-> DataLoader:
        """ generate the dataloader for differnet datasets, which is used 
        for training or evaluating.

        Args:
            dataset (PytorchDataset): [description]
            part (str): [description]
            metadata (MetaData): [description]

        Returns:
            torch.utils.data.DataLoader: [description]
        """        
        return DataLoader(getattr(dataset,part), batch_size=metadata.batch_size, shuffle=True, num_workers=0,drop_last=True)
    
    


PYTORCH_DATASET_HANDLERS={
    ListPytorchDatasetHandlers.StdPytorchDatasetHandler: StdPytorchDatasetHandler,
    ListPytorchDatasetHandlers.PytorchConv1dDatasetHandler: PytorchConv1dDatasetHandler,
    ListPytorchDatasetHandlers.PytorchUxyzDatasetHandler: PytorchUxyzDatasetHandler,
    
}

def reshape_4_1Dconv(x:np.ndarray, channel:int=1)-> np.ndarray:
    """[summary]

    Args:
        x (np.ndarray): [description]
        channel (int, optional): [description]. Defaults to 1.

    Returns:
        np.ndarray: [description]
    """
    
    return x.reshape([-1, channel, x.shape[1]])


if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    





        

        