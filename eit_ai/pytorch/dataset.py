from typing import Union
import numpy as np
import random
from sklearn import model_selection
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sklearn.model_selection

import os
import sys
sys.path.append('./')

from eit_ai.train_utils.dataset import Datasets, XYSet, scale_prepocess

from eit_ai.train_utils.metadata import MetaData
from logging import getLogger

logger = getLogger(__name__)

class StdPytorchDataset(Datasets):
   
    def get_X(self, part:str='train'):
        return getattr(self, part).get_set()[0]

    def get_Y(self, part:str='train'):
        return getattr(self, part).get_set()[1]

    def get_samples(self, part: str):
        return getattr(self, part).get_set()

    def _preprocess(
        self,
        X:np.ndarray,
        Y:np.ndarray,
        metadata:MetaData)->tuple[Union[np.ndarray,None],Union[np.ndarray,None]]:
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
        x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y,test_size=self._test_ratio)
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp, test_size=self._val_ratio)
        
        self._idx_train= x_train[:,-1].tolist()
        self._idx_val= x_val[:,-1].tolist()
        self._idx_test= x_test[:,-1].tolist()
        metadata.set_idx_samples(self._idx_train, self._idx_val, self._idx_test)

        self.train=XYSet(x=x_train[:,:-1], y=y_train)
        self.val=XYSet(x=x_val[:,:-1], y=y_val)
        self.test=XYSet(x=x_test[:,:-1], y=y_test)

    def _mk_dataset_from_indexes(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """rebuild the dataset with the indexes """
        self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
        self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
        self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
        self.train=TorchDataset(x=X[self._idx_train,:], y=Y[self._idx_train,:])
        self.val=TorchDataset(x=X[self._idx_val,:], y=Y[self._idx_val,:])
        self.test=TorchDataset(x=X[self._idx_test,:], y=Y[self._idx_test,:])

class TorchDataset(Dataset):

    def __init__(self, x, y, loaded_data:np.ndarray, metadata: MetaData):
 
        self.data = loaded_data
        self.X = torch.Tensor(x[:, :-1]).float()
        self.Y = torch.Tensor(y[:, [-1]]).float()
        # self.idx= torch.Tensor(np.array(range(loaded_data.shape[0]))).float()
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index:Union[int, list[int]]=None)->tuple[torch.Tensor,torch.Tensor]:
        # idx = np.reshape(range(self.X.shape[0]), (self.X.shape[0], 1))
        # X = np.concatenate((self.X, idx), axis=1)
        # x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=self._test_ratio)
        # x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp,
        #                                                                           test_size=self._val_ratio)

        # self._idx_train = x_train[:, -1].tolist()
        # self._idx_val = x_val[:, -1].tolist()
        # self._idx_test = x_test[:, -1].tolist()
        # self.metadata.set_idx_samples(self._idx_train, self._idx_val, self._idx_test)
        
        # self.train = XYSet(x=x_train[:, :-1], y=y_train)
        # self.val = XYSet(x=x_val[:, :-1], y=y_val)
        # self.test = XYSet(x=x_test[:, :-1], y=y_test)
        return self.X[index], self.Y[index] if index else self.X, self.Y

    
    def get_idx(self)-> list[int]:
        return self.idx.tolist()

# class STdtorchDataset(Datasets):
#     X= None
#     Y= None

#     def __init__(
#         self,
#         loaded_data:np.ndarray= None,
#         idx:list[int]=None,
#         dataset:TorchDatasetWithIdx= None)-> None:
#         """[summary]

#         Args:
#             loaded_data (np.ndarray, optional): [description]. Defaults to None.
#             idx (list[int], optional): [description]. Defaults to None.
#             dataset (TorchDatasetWithIdx, optional): [description]. Defaults to None.
#         """        
       
        
#         if loaded_data:
#             self.data = loaded_data
#             if idx:
#                 self.X = torch.Tensor(loaded_data[idx, :-1]).float()
#                 self.Y = torch.Tensor(loaded_data[idx, [-1]]).float()
#             else:
#                 self.X = torch.Tensor(loaded_data[:, :-1]).float()
#                 self.Y = torch.Tensor(loaded_data[:, [-1]]).float()

#         # self.X = X
#         # self.Y = Y
#         # self.X = loaded_data[:, :-1]
#         # self.Y = loaded_data[:,[-1]]

#     def set_from_dataset(self):
#         if dataset:
#             self.X = dataset.X
#             self.Y = dataset.Y

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index)->tuple[torch.Tensor,torch.Tensor]:
#         return self.X[index], self.Y[index]



# def _mk_Dataloader(loaded_data, dataset_type)-> DataLoader:
#     loaded_data = TorchDatasetWithIdx(loaded_data)
#     train_size = int(len(loaded_data) * 0.6)
#     val_size = int(len(loaded_data) * 0.2)
#     test_size = int(len(loaded_data) * 0.2)

#     train_set, val_set, test_set = torch.utils.data.random_split(
#         loaded_data, [train_size, val_size, test_size])

#     _idx_train= train_set.get_idx()
#     train= STdtorchDataset(dataset=train_set)

#     train=STdtorchDataset(loaded_data=loaded_data, idx=_idx_train, dataset=train_set)
   
#     if(dataset_type == 'train_set'):
#         return DataLoader(train, batch_size=5, shuffle=True, num_workers=0)
#     if(dataset_type == 'val_set'):
#          return DataLoader(val_set, batch_size=5, shuffle=False, num_workers=0)
#     if(dataset_type == 'test_set'):
#          return DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)
        


if __name__ == "__main__":
    # from glob_utils.log.log  import change_level_logging, main_log
    # import logging
    # main_log()
    # change_level_logging(logging.DEBUG)

    # X = np.array([[random.randint(0, 100) for _ in range(4)] for _ in range(100)])
    # Y = np.array([random.randint(0, 100) for _ in range(100)])
    # print(f'{X}; {X.shape}\n; {Y}; {Y.shape}')
    
    X = np.random.randn(100, 4)
    Y = np.random.randn(100)

    XY = np.concatenate((X, Y[:, np.newaxis]), axis=1)

    # create the normalized dataset
    rdn_dataset = TorchDatasetWithIdx(XY)

#     for i in range(len(rdn_dataset)):
#         print(rdn_dataset[i])

# train_size = int(len(rdn_dataset) * 0.6)
# val_size = int(len(rdn_dataset) * 0.2)
# test_size = int(len(rdn_dataset) * 0.2)

# train_set, val_set, test_set = torch.utils.data.random_split(rdn_dataset, [train_size, val_size, test_size])

# train_loader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=0)

    # train_loader = _mk_Dataloader(XY_normal, 'train_set')

    # class Model(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.layers = nn.Sequential(nn.Linear(4, 3),
    #                                     nn.BatchNorm1d(3),
    #                                     nn.ReLU(),
    #                                     nn.Linear(3, 1)
    #         )
            
    #     def forward(self, x):
    #         return self.layers(x)


    # net = Model() # self.model

    # loss_mse = nn.MSELoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # for epoch in range(10):
    #     for i, data in enumerate(train_loader, 0):
    #         inputs, labels = data

    #         y_pred = net(inputs)
    #         loss = loss_mse(y_pred, labels)
    #         print(epoch, i, loss.item())

    #         optimizer.zero_grad()
    #         loss.backward()

    #         optimizer.step()





        

        