

import os
import logging

import numpy as np
import tensorflow as tf
from eit_ai.train_utils.dataset import (AiDatasetHandler, StdAiDatasetHandler,
                                          scale_preprocess)
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.lists import ListKerasDatasetHandlers
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

################################################################################
# Dataset for Keras Model using the tf.data.Dataset class
################################################################################
class KerasDatasetHandler(StdAiDatasetHandler):
    """ Identical to StdAiDatasetHandler
    """    
class TfDatasetHandler(StdAiDatasetHandler):
    def _post_init(self):
        """
        """  
        raise NotImplementedError()     
   
    # def get_X(self, part:str='train'):
    #     return getattr(self, part)

    # def get_Y(self, part:str='train'):
    #     return None
    
    # def get_samples(self, part: str):
    #     x=[]
    #     y=[]
    #     for i, xy in enumerate(getattr(self, part)):
    #         # print('#', i, eval_dataset.batch_size,eval_dataset.test_len)
    #         if self._batch_size:
    #             if (i+1)*self._batch_size>self._test_len:
    #                 break
    #             x.append(xy[0].numpy())
    #             y.append(xy[1].numpy())
    #         else:
    #             #print(xy[elem_idx], xy[elem_idx].shape)
    #             x.append(xy[0].numpy().reshape(xy[0].shape[0],1).T)
    #             y.append(xy[1].numpy().reshape(xy[1].shape[0],1).T)
                
    #     samples_x = np.concatenate(x, axis=0)
    #     samples_y = np.concatenate(y, axis=0)
    #     return samples_x, samples_y

    # def _preprocess(self, X, Y, metadata:MetaData):
    #     """return X, Y preprocessed"""
    #     X=scale_prepocess(X, metadata.normalize[0])
    #     Y=scale_prepocess(Y, metadata.normalize[0])
    #     return X, Y

    # def _mk_dataset(self, X, Y, metadata:MetaData)-> None:
    #     """build the dataset"""
    #     real_data= tf.data.Dataset.from_tensor_slices((X, Y))
    #     indexes = tf.data.Dataset.from_tensor_slices(tf.range(self._nb_samples))
    #     samples = tf.data.Dataset.zip((real_data, indexes))
    #     #samples=samples.shuffle()

    #     self._train_len=int((1-self._test_ratio-self._val_ratio)*self._nb_samples)
    #     self._val_len=int(self._val_ratio*self._nb_samples)
    #     self._test_len= int(self._test_ratio*self._nb_samples)

    #     train_tmp= samples.take(self._train_len)
    #     val_test_tmp= samples.skip(self._train_len)
    #     val_tmp=  val_test_tmp.take(self._val_len)
    #     test_tmp=  val_test_tmp.skip(self._val_len)

    #     idx=train_tmp.map(lambda xy, idx: idx)
    #     self._idx_train= np.array(list(idx.as_numpy_iterator()))
    #     train_tmp=train_tmp.map(lambda xy, idx: xy)
        
    #     idx=val_tmp.map(lambda xy, idx: idx)
    #     self._idx_val= np.array(list(idx.as_numpy_iterator()))
    #     val_tmp=val_tmp.map(lambda xy, idx: xy)

    #     idx=test_tmp.map(lambda xy, idx: idx)
    #     self._idx_test= np.array(list(idx.as_numpy_iterator()))
    #     test_tmp=test_tmp.map(lambda xy, idx: xy)

    #     scaler = MinMaxScaler()

    #     # transform data
    #     train_tmp=train_tmp.map(lambda x, y: (scaler.fit_transform(x), scaler.fit_transform(y)))
    #     val_tmp=val_tmp.map(lambda x, y: (scaler.fit_transform(x), scaler.fit_transform(y)))
    #     val_tmp=val_tmp.map(lambda x, y: (scaler.fit_transform(x), scaler.fit_transform(y)))
        

    #     if self._batch_size:
    #         self.train= train_tmp.repeat().batch(self._batch_size)
    #         self.val= val_tmp.repeat().batch(self._batch_size)
    #         self.test=test_tmp.repeat().batch(self._batch_size)
    #     else:
    #         self.train= train_tmp
    #         self.val= val_tmp
    #         self.test=test_tmp

    # def _mk_dataset_from_indexes(self, X, Y, metadata:MetaData)-> None:
    #     """rebuild the dataset with the indexes """
    #     # self._idx_train= convert_vec_to_int(metadata.idx_samples['idx_train'])
    #     # self._idx_val= convert_vec_to_int(metadata.idx_samples['idx_val'])
    #     # self._idx_test= convert_vec_to_int(metadata.idx_samples['idx_test'])   
    #     # self.train=XYSet(x=X[self._idx_train,:], y=Y[self._idx_train,:])
    #     # self.val=XYSet(x=X[self._idx_val,:], y=Y[self._idx_val,:])
    #     # self.test=XYSet(x=X[self._idx_test,:], y=Y[self._idx_test,:])

    # # def mk_tf_dataset(self, X, Y, metadata:MetaData):
    # #     self.use_tf_dataset= True       
    # #     self.set_sizes_dataset(X, Y, metadata)

################################################################################
# Keras Datasets
################################################################################
""" Dictionary listing all Keras datasets available
"""
KERAS_DATASET_HANDLERS={
    ListKerasDatasetHandlers.KerasDatasetHandler: KerasDatasetHandler,
    ListKerasDatasetHandlers.TfDatasetHandler: TfDatasetHandler
}







if __name__ == "__main__":

    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    path= "E:/EIT_Project/05_Engineering/04_Software/Python/eit_ai/datasets/DStest/test10_infos2py.mat" 
    print(os.path.split(os.path.split(path)[0]))
