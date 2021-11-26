




# from abc import ABC, abstractmethod
# from enum import Enum
# from logging import getLogger

# from eit_ai.train_utils.metadata import MetaData 
# from eit_ai.train_utils.dataset import DeepLDataset
# import numpy as np
# import sklearn.model_selection
# from sklearn.preprocessing import MinMaxScaler

# logger = getLogger(__name__)

# ################################################################################
# # XY Set Class for Custom standard dataset
# ################################################################################

# class XYSet(object):
#     x=np.array([])
#     y = np.array([])
#     def __init__(self,x=np.array([]), y=np.array([])) -> None:
#         super().__init__()
#         self.set_data(self, x, y)
 
#     def set_data(self, x, y):
#         self.x=x
#         self.y=y

#     def get_set(self):
#         return self.x, self.y
    
# ################################################################################
# # Custom standard dataset
# ################################################################################

# class StdDataset(DeepLDataset):
   
#     def get_X(self, part:str='train'):
#         return getattr(self, part).get_set()[0]

#     def get_Y(self, part:str='train'):
#         return getattr(self, part).get_set()[1]

#     def get_samples(self, part: str):
#         return getattr(self, part).get_set()

#     def _preprocess(self, X, Y, metadata:MetaData):
#         """return X, Y preprocessed"""
#         X=scale_prepocess(X, metadata.normalize[0])
#         Y=scale_prepocess(Y, metadata.normalize[0])
#         return X, Y

#     def _mk_dataset(self, X, Y, metadata:MetaData)-> None:
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

#     def _mk_dataset_from_indexes(self, X, Y, metadata:MetaData)-> None:
#         """rebuild the dataset with the indexes """
#         self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
#         self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
#         self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
#         self.train=XYSet(x=X[self._idx_train,:], y=Y[self._idx_train,:])
#         self.val=XYSet(x=X[self._idx_val,:], y=Y[self._idx_val,:])
#         self.test=XYSet(x=X[self._idx_test,:], y=Y[self._idx_test,:])

# ################################################################################
# # Methods
# ################################################################################

# def scale_prepocess(x, scale:bool=True):
#     if scale:
#         scaler = MinMaxScaler()
#         x= scaler.fit_transform(x)
#     return x

# def convert_to_int(x):
#     return np.int(x)
# convert_vec_to_int = np.vectorize(convert_to_int)

# class ListDatasets(Enum):
#     StdDataset=['StdDataset', StdDataset]
#     """"""

if __name__ == "__main__":
    from eit_ai.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)

    