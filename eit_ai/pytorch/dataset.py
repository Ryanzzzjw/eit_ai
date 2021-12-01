import random
from logging import getLogger
from typing import Union

import numpy as np
import sklearn.model_selection
import torch
from eit_ai.train_utils.dataset import (Datasets, convert_vec_to_int,
                                        scale_prepocess)
from eit_ai.train_utils.lists import PytorchDatasets
from eit_ai.train_utils.metadata import MetaData
from sklearn import model_selection
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = getLogger(__name__)

class StdPytorchDataset(Datasets):
   
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

        self.train=TorchDataset(x=x_train[:,:-1], y=y_train)
        self.val=TorchDataset(x=x_val[:,:-1], y=y_val)
        self.test=TorchDataset(x=x_test[:,:-1], y=y_test)
        # gen=DataloaderGenerator()
        # self.train=gen.make(dataset=train,metadata=metadata)
        # self.val=gen.make(dataset=val,metadata=metadata)
        # self.test=gen.make(dataset=test,metadata=metadata)
        


    def _mk_dataset_from_indexes(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """rebuild the dataset with the indexes """
        self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
        self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
        self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
        self.train=TorchDataset(x=X[self._idx_train,:], y=Y[self._idx_train,:])
        self.val=TorchDataset(x=X[self._idx_val,:], y=Y[self._idx_val,:])
        self.test=TorchDataset(x=X[self._idx_test,:], y=Y[self._idx_test,:])
        # gen=DataloaderGenerator()
        # self.train=gen.make(dataset=train,metadata=metadata)
        # self.val=gen.make(dataset=val,metadata=metadata)
        # self.test=gen.make(dataset=test,metadata=metadata)

class TorchDataset(Dataset):

    def __init__(self, x:np.ndarray, y:np.ndarray)-> None:
        """[summary]

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
        return len(self.X.shape[0])

    def __getitem__(self, index:Union[int, list[int]]=None)->tuple[torch.Tensor,torch.Tensor]:
        
        return self.X[index], self.Y[index]
        
    def get_set(self)->tuple[np.ndarray,np.ndarray]:
        return self.X, self.Y

class DataloaderGenerator(object):
    def make(self, dataset:TorchDataset, metadata:MetaData)->DataLoader:
        # self.train = StdPytorchDataset().train
        # self.val = StdPytorchDataset().val
        # self.test = StdPytorchDataset().test
        return DataLoader(dataset, batch_size=metadata.batch_size, shuffle=True, num_workers=0)
class DataloaderGenerator_old(object):
    def make(self, dataset:StdPytorchDataset, part:str, metadata:MetaData)->DataLoader:
        # self.train = StdPytorchDataset().train
        # self.val = StdPytorchDataset().val
        # self.test = StdPytorchDataset().test
        return DataLoader(getattr(dataset,part), batch_size=metadata.batch_size, shuffle=True, num_workers=0)
    # def _mk_dataloader(self):
    #     self.trainLoader = DataLoader(self.train, batch_size=MetaData.batch_size, shuffle=True, num_workers=0)
    #     self.valLoader = DataLoader(self.val, batch_size=MetaData.batch_size, shuffle=True, num_workers=0)
    #     self.trainLoader = DataLoader(self.train, batch_size=MetaData.batch_size, shuffle=True, num_workers=0)


PYTORCH_DATASETS={
    PytorchDatasets.StdDataset: StdPytorchDataset
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

    XY = np.concatenate((X, Y[:, np.newaxis]), axis=1)

    
    rdn_dataset = StdPytorchDataset()
    trainLoader = DataloaderGenerator(rdn_dataset.train)


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





        

        