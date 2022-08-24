
from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import Any, Union

import numpy as np
import sklearn.model_selection
from eit_ai.raw_data.raw_samples import RawSamples
from eit_ai.train_utils.lists import ListNormalizations, get_from_dict
from eit_ai.train_utils.metadata import MetaData
from scipy.stats import zscore
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

logger = logging.getLogger(__name__)
################################################################################
# Custom Exeptions/Errors for Dataset
################################################################################
class WrongSingleXError(Exception):
    """"""
################################################################################
# Abstract Class for AiDatasetHandler
################################################################################

class AiDatasetHandler(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_cls=SimpleDataset
        self.train:AiDataset=None
        self.val:AiDataset=None
        self.test:AiDataset=None
        self._nb_samples:int= 0
        self._batch_size:int=32
        self._test_ratio:float=0.20
        self._val_ratio:float=0.2
        self._train_len:int= 0
        self._val_len:int= 0
        self._test_len:int= 0
        self._idx_train:list=[]
        self._idx_val:list=[]
        self._idx_test:list= []
        self.src_file:str= ''
        self.input_size:int= 0
        self.ouput_size:int= 0
        
        self.fwd_model:dict={}
        self.sim:dict={}
        
        self._post_init()

    def set_dataset_type(self, metadata:MetaData):
        """[summary]
        """        

    def build(self,raw_samples:RawSamples, metadata:MetaData):
        """

        Args:
            raw_samples (RawSamples): [description]
                self.X: ArrayLike(n_samples, n_features)
                self.Y: ArrayLike(n_samples, n_labels)
            metadata (MetaData): [description]
        """        
        X=raw_samples.X
        Y=raw_samples.Y
        self.fwd_model= raw_samples.fwd_model
        self.sim= raw_samples.sim
        metadata._nb_samples= raw_samples.nb_samples

        self._set_sizes_dataset(X, Y, metadata)
        X, Y= self._preprocess(X, Y, metadata)
        if self._is_indexes(metadata):
            self._mk_dataset_from_indexes(X, Y, metadata)
        else:
            self._mk_dataset(X, Y, metadata)
        self._compute_sizes()
        metadata._nb_samples= self._nb_samples
        metadata._train_len=self._train_len
        metadata._val_len=self._val_len
        metadata._test_len=self._test_len
        metadata.input_size=self.input_size
        metadata.output_size=self.ouput_size

    def _set_sizes_dataset(self,X:np.ndarray, Y:np.ndarray, metadata:MetaData):
        """[summary]

        Args:
            X ([type]): [description]
            Y ([type]): [description]
            metadata (MetaData): [description]
        """ 
        self._batch_size = metadata.batch_size
        self._test_ratio= metadata.test_ratio
        self._val_ratio = metadata.val_ratio
        

    def _compute_sizes(self):
        
        # logger.debug(f'Size of X and Y: {X.shape=}, {Y.shape=}')       
        # if metadata._nb_samples != self._nb_samples:
        #     raise TypeError(f'wrong shape {X=}, {X.shape=}; {Y=}, {Y.shape=}')
        
        # self.input_size= self.get_X('train').shape[1]
        self.input_size, self.ouput_size= self.train.get_inout_sizes()
        # self.ouput_size= self.get_Y('train').shape[1]
        
        self._train_len=len(self.train)
        self._val_len=len(self.val)
        self._test_len= len(self.test)
        self._nb_samples= self._train_len + self._val_len + self._test_len
        
        logger.info(f'{self._nb_samples=}')
        logger.info(f'{self.ouput_size=}')
        logger.info(f'{self.input_size=}')
        logger.info(f'Length of train: {self._train_len}')
        logger.info(f'Length of val: {self._val_len}')
        logger.info(f'Length of test: {self._test_len}')

    def _is_indexes(self, metadata:MetaData):
        """[summary]

        Args:
            metadata (MetaData): [description]

        Returns:
            [type]: [description]
        """
        return metadata.idx_samples is not None
    
    def format_single_X(self, single_X:np.ndarray, metadata:MetaData, preprocess:bool=False)->np.ndarray:

        formated_X= single_X.flatten()

        if formated_X.shape[0] != self.input_size:
            raise WrongSingleXError(
                f'{single_X=}\n {formated_X.shape[0]=} != {self.input_size}')

        formated_X= np.reshape(formated_X,(1, self.input_size))
        logger.debug(f'{formated_X=}, {formated_X.shape=}')
        if not preprocess:
            if metadata.model_type == 'Conv1dNet':
                formated_X= np.reshape(formated_X,(-1, 1, self.input_size))
            return formated_X
        prepro_X= self._preprocess(formated_X, None, metadata)[0]
        if metadata.model_type == 'Conv1dNet':
            prepro_X= np.reshape(prepro_X,(-1, 1, self.input_size))
        logger.debug(f'{prepro_X=}, {prepro_X.shape=}')
        return prepro_X

    @abstractmethod    
    def _post_init(self):
        """allow different init conditions
        """        
    @abstractmethod
    def get_X(self, part:str='train')->np.ndarray:
        """return X from a dataset part (train, val, test)"""

    @abstractmethod
    def get_Y(self, part:str='train')->np.ndarray:
        """return Y from a dataset part (train, val, test)"""

    @abstractmethod
    def get_samples(self, part: str)->tuple[np.ndarray,np.ndarray]:
        """Return all samples_x, and samples_y as a tuple"""

    @abstractmethod
    def _preprocess(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)->tuple[Union[np.ndarray,None],Union[np.ndarray,None]]:
        """return X, Y preprocessed"""

    @abstractmethod
    def _mk_dataset(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """build the dataset"""

    @abstractmethod
    def _mk_dataset_from_indexes(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """rebuild the dataset with the indexes """


################################################################################
# AiDataset Abstract class
################################################################################

class AiDataset(ABC):
    """Dataset abstract class

    should contain x and y of the dataset
    also the creator should be AiDataset(x, y)
    """    
    
    

    @abstractmethod
    def __len__(self):
        """"""
        
    @abstractmethod
    def get_inout_sizes(self):
        """"""
        
    @abstractmethod
    def get_set(self)->tuple[np.ndarray,np.ndarray]:
        """ Return the x and y of the dataset
        Returns:
            tuple[np.ndarray,np.ndarray]: (x,y)
        """

################################################################################
# Custom standard dataset
################################################################################

class SimpleDataset(AiDataset):
    """Simple Dataset containing a x and a y as ndarray
    """    
    _x:np.ndarray=np.array([])
    _y:np.ndarray=np.array([])

    def __init__(self,x:np.ndarray, y:np.ndarray) -> None:
        super().__init__()
        self._x=x
        self._y=y
        
    def __len__(self):
        return len(self._x)
    
    def get_inout_size(self):
        return self._x.shape[1], self._y.shape[1]

    def __len__(self):
        return len(self._x)
    
    def get_inout_sizes(self):
        return self._x.shape[1], self._y.shape[1]
    
    def get_set(self)->tuple[np.ndarray,np.ndarray]:
        return self._x, self._y
    
    
################################################################################
# Custom standard Datasethandler
################################################################################

class StdAiDatasetHandler(AiDatasetHandler):
  
    def _post_init(self):
        self.dataset_cls= SimpleDataset 

    def get_X(self, part:str='train')->np.ndarray:
        return getattr(self, part).get_set()[0]

    def get_Y(self, part:str='train')->np.ndarray:
        return getattr(self, part).get_set()[1]

    def get_samples(self, part: str)->tuple[np.ndarray,np.ndarray]:
        return getattr(self, part).get_set()

    def _preprocess(
        self,
        X:np.ndarray,
        Y:np.ndarray,
        metadata:MetaData
        )->tuple[Union[np.ndarray,None],Union[np.ndarray,None]]:
        """return X, Y preprocessed"""
        if isinstance(metadata.normalize[0], bool):
            X=scale_preprocess(X, metadata.normalize[0])
            Y=scale_preprocess(Y, metadata.normalize[1])
        else:
            X= get_from_dict(
                metadata.normalize[0],NORMALIZATIONS,ListNormalizations)(X)
            Y= get_from_dict(
                metadata.normalize[1],NORMALIZATIONS,ListNormalizations)(Y)
        log_msg='Preprocessing - Done :'
        if Y is not None:
            logger.info(f'{log_msg} {X.shape=}\n, {Y.shape=}') 
            logger.debug(f'{log_msg} {X=}\n, {Y=}') 
        else:
            logger.info(f'{log_msg} {X.shape=}')
            logger.debug(f'{log_msg} {X=}')
        return X, Y

    def _mk_dataset(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """build the dataset"""
        idx=np.reshape(range(X.shape[0]),(X.shape[0],1))
        X= np.concatenate(( X, idx ), axis=1)
        x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y,test_size=self._test_ratio,random_state=42)
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp, test_size=self._val_ratio,random_state=42)
        
        self._idx_train= x_train[:,-1].tolist()
        self._idx_val= x_val[:,-1].tolist()
        self._idx_test= x_test[:,-1].tolist()
        metadata.set_idx_samples(self._idx_train, self._idx_val, self._idx_test)

        self.train=self.dataset_cls(x_train[:,:-1], y_train)
        self.val=self.dataset_cls(x_val[:,:-1], y_val)
        self.test=self.dataset_cls(x_test[:,:-1], y_test)

    def _mk_dataset_from_indexes(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
        """rebuild the dataset with the indexes """
        self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
        self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
        self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
        self.train=self.dataset_cls(X[self._idx_train,:], Y[self._idx_train,:])
        self.val=self.dataset_cls(X[self._idx_val,:], Y[self._idx_val,:])
        self.test=self.dataset_cls(X[self._idx_test,:], Y[self._idx_test,:])

################################################################################
# Methods
################################################################################

def convert_to_int(x:Any)->int:
    return np.int(x)

convert_vec_to_int = np.vectorize(convert_to_int)

def scale_preprocess(x:np.ndarray, scale:bool=True)->Union[np.ndarray,None]:
    """Normalize input x using minMaxScaler 
    Attention: if x.shape is (1,n) it wont work

    Args:
        x (np.ndarray): array-like of shape (n_samples, n_features)
                        Input samples.
        scale (bool, optional):. Defaults to True.

    Returns:
        np.ndarray: ndarray array of shape (n_samples, n_features_new)
            Transformed array.
    """    
    if scale:
        scaler = MinMaxScaler()
        # scaler = MaxAbsScaler()
        x= scaler.fit_transform(x.T).T if x is not None else None
        # logger.debug(f'{scaler.scale_=}, {scaler.scale_.shape=}')
    return x
################################################################################
# Preprocessing methods
################################################################################
def _preprocess_identity(x:np.ndarray)->np.ndarray:
    """Preprocessing returning indentity of x  
    
    Args:
        x (np.ndarray): array-like of shape (n_samples, n_features)
                        Input samples.

    Returns:
        np.ndarray: ndarray array of shape (n_samples, n_features_new)
            Transformed array.
    """    
    x= preprocess_guard(x)
    return x if x.size > 0 else x

def _preprocess_zscore(x:np.ndarray)->np.ndarray:
    """Preprocessing returning zscore of x  

    Args:
        x (np.ndarray): array-like of shape (n_samples, n_features)
                        Input samples.

    Returns:
        np.ndarray: ndarray array of shape (n_samples, n_features_new)
            Transformed array.
    """
    x= preprocess_guard(x)    
    return zscore(x, axis=1) if x.size > 0 else x

def _preprocess_minmax_01(x:np.ndarray)->Union[np.ndarray,None]:
    """Preprocessing returning MinMax of x with scaling range [0,1]

    Args:
        x (np.ndarray): array-like of shape (n_samples, n_features)
                        Input samples.

    Returns:
        np.ndarray: ndarray array of shape (n_samples, n_features_new)
            Transformed array.
    """    
    x= preprocess_guard(x)
    if x.size== 0:
        return x
    scaler = MinMaxScaler(feature_range=(0,1))
    x=scaler.fit_transform(x.T).T
    logger.debug(f'{scaler.scale_=}, {scaler.scale_.shape=}')
    return x

def _preprocess_minmax_11(x:np.ndarray)->np.ndarray:
    """Preprocessing returning MinMax of x with scaling range [-1,1]

    Args:
        x (np.ndarray): array-like of shape (n_samples, n_features)
                        Input samples.

    Returns:
        np.ndarray: ndarray array of shape (n_samples, n_features_new)
            Transformed array.
    """    
    x= preprocess_guard(x)
    if x.size== 0:
        return x
    scaler = MinMaxScaler(feature_range=(-1,1))
    x=scaler.fit_transform(x.T).T
    logger.debug(f'{scaler.scale_=}, {scaler.scale_.shape=}')
    return x

def preprocess_guard(x:np.ndarray)->np.ndarray:
    """Check if x is a 2d `ndarray`
    
    if x ==`None`>> convert to 2d array

    Args:
        x (np.ndarray): a 2d `ndarray`

    Raises:
        TypeError: raise if x is not a 2d `ndarray`

    Returns:
        np.ndarray: x or empty 2d array if x is `None`
    """    
    if x is None:
        return np.array([[]])
    if not isinstance(x, np.ndarray) or x.ndim !=2:
        raise TypeError(f'x is not an 2d ndarray{x=}')
    return x

NORMALIZATIONS={
    ListNormalizations.Identity:_preprocess_identity,
    ListNormalizations.MinMax_01:_preprocess_minmax_01,
    ListNormalizations.MinMax_11:_preprocess_minmax_11,
    ListNormalizations.Norm:_preprocess_zscore
}


if __name__ == "__main__":
    import logging
    import random

    from glob_utils.log.log import change_level_logging, main_log
    from matplotlib import pyplot as plt
    main_log()
    change_level_logging(logging.DEBUG)
    if not np.array([[]]):
        print('array empty')


    data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    scaler = MinMaxScaler()
    print(f'{data=}')
    print(scaler.fit(data))
    print(scaler.data_max_)
    print(scaler.transform(data))
    print(scaler.transform([[2, 2]]))

    rge=4
    x = np.array([[random.random() * (row + 1) + 2 for _ in range(100)] for row in range(rge)])

    print(f'{x=}, {x.shape=}')

    for idx in range(rge):
        plt.figure()
        plt.plot(x[idx], label='x')
        # plt.plot(_prepocess_identity(x)[idx],label='ident(x)')
        plt.plot(_preprocess_minmax_11(x)[idx],label='mm-11(x)')
        plt.plot(_preprocess_minmax_01(x)[idx],label='mm01(x)')
        plt.plot(_preprocess_zscore(x)[idx],label='zscore(x)')



        plt.legend()
        plt.figure()
        plt.boxplot([
            x[idx],
            _preprocess_minmax_11(x)[idx],
            _preprocess_minmax_01(x)[idx],
            _preprocess_zscore(x)[idx]],labels=['x','mm-11(x)','mm01(x)','zscore(x)'])
        plt.legend()
        # plt.show(block=False)
    change_level_logging(logging.INFO)
    plt.show()

