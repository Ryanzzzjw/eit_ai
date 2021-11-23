from typing import Union
import numpy as np
import random
from eit_tf_workspace.train_utils.dataset import Datasets, scale_prepocess

from eit_tf_workspace.train_utils.metadata import MetaData
from logging import getLogger

logger = getLogger(__name__)

# class StdPytorchDataset(Datasets):
   
#     def get_X(self, part:str='train'):
#         return getattr(self._pytorch_dataset, part)[0]

#     def get_Y(self, part:str='train'):
#         return getattr(self, part).get_set()[1]

#     def get_samples(self, part: str):
#         return getattr(self, part).get_set()

#     def _preprocess(
#         self,
#         X:np.ndarray,
#         Y:np.ndarray,
#         metadata:MetaData)->tuple[Union[np.ndarray,None],Union[np.ndarray,None]]:
#         """return X, Y preprocessed"""
#         self._pytorch_dataset=
#         X=scale_prepocess(X, metadata.normalize[0])
#         Y=scale_prepocess(Y, metadata.normalize[1])
#         if Y is not None:
#             logger.debug(f'Size of X and Y (after preprocess): {X.shape=}, {Y.shape=}')     
#         else:
#             logger.debug(f'Size of X (after preprocess): {X.shape=}')
#         return X, Y

#     def _mk_dataset(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
#         """build the dataset"""
        
        
#         idx=np.reshape(range(X.shape[0]),(X.shape[0],1))
#         X= np.concatenate(( X, idx ), axis=1)
#         x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y,test_size=self._test_ratio)
#         x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp, test_size=self._val_ratio)
        
#         self._idx_train= x_train[:,-1].tolist()
#         self._idx_val= x_val[:,-1].tolist()
#         self._idx_test= x_test[:,-1].tolist()
#         metadata.set_idx_samples(self._idx_train, self._idx_val, self._idx_test)

#         self.train=XYSet(x=x_train[:,:-1], y=y_train)
#         self.val=XYSet(x=x_val[:,:-1], y=y_val)
#         self.test=XYSet(x=x_test[:,:-1], y=y_test)

#     def _mk_dataset_from_indexes(self, X:np.ndarray, Y:np.ndarray, metadata:MetaData)-> None:
#         """rebuild the dataset with the indexes """
#         self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
#         self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
#         self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
#         self.train=XYSet(x=X[self._idx_train,:], y=Y[self._idx_train,:])
#         self.val=XYSet(x=X[self._idx_val,:], y=Y[self._idx_val,:])
#         self.test=XYSet(x=X[self._idx_test,:], y=Y[self._idx_test,:])



if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    X=np.array([ [random.randint(0,100) for _ in range(4)] for _ in range(100)])
    Y=np.array([ random.randint(0,100) for _ in range(100)])

    print(f'{X=}; {X.shape=}\n, {Y=}; {Y.shape=}')


