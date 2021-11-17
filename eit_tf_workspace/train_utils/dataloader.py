# is responsible of loading the dataset used for training und evaluation
# - from raw samples
# - from metadata file


import os
from logging import getLogger

import numpy as np
from scipy.io.matlab.mio import savemat

import eit_tf_workspace.constants as const
from eit_tf_workspace.train_utils.dataset import Datasets, scale_prepocess
from eit_tf_workspace.raw_data.matlab import (LoadCancelledException, MatlabSamples,
                                               load_predictions_EIDORS)
from eit_tf_workspace.raw_data.raw_samples import RawSamples
from eit_tf_workspace.utils.path_utils import (get_date_time, save_as_pickle,
                                         save_as_txt)
from eit_tf_workspace.train_utils.metadata import MetaData

logger = getLogger(__name__)

# def load_samples( raw_samples:RawSamples, src_path:str, metadata:MetaData)-> None:
#     """"""
#     raw_samples.load(
#         file_path=src_path,
#         nb_samples2load=metadata.nb_samples,
#         data_sel= metadata.data_select)
#     metadata.set_dataset_src_file(raw_samples.file_path)

# def reload_samples(raw_samples:RawSamples, dir_path:str='')-> None:
#     """"""
#     metadata=MetaData()
#     metadata.reload(dir_path)

#     raw_samples.load(
#         file_path=metadata.dataset_src_file[0],
#         nb_samples2load=metadata.nb_samples,
#         data_sel= metadata.data_select)

# def build_dataset(src_path:str, metadata:MetaData)-> Datasets:
#     """"""

#     raw_data=get_matlab_dataset(
#         file_path=src_path,
#         data_sel= metadata.data_select,
#         nb_samples2load=metadata.nb_samples
#     )
#     metadata.set_dataset_src_file(raw_data.file_path)

#     dataset = create_dataset(raw_data, metadata)
#     save_idx_samples_2matfile(metadata)
#     return dataset

# def reload_matlab_dataset(dir_path:str=''):
#     """"""
#     metadata=MetaData()
#     metadata.reload(dir_path)

#     raw_data=get_matlab_dataset(
#         file_path=metadata.dataset_src_file[0],
#         data_sel= metadata.data_select,
#         nb_samples2load=metadata.nb_samples)

#     dataset = create_dataset(raw_data, metadata)
      
#     return metadata, dataset

# def create_dataset(
#         raw_data:MatlabDataSet,
#         metadata:MetaData=None) -> Datasets:
#     """"""
    

#     return dataset
    
# def extract_samples(dataset:Datasets, dataset_part='test', idx_samples=None):
    
#     samples_x, samples_y= dataset.get_samples(part=dataset_part)

#     # if dataset.use_tf_dataset:
        
#     #     for i, xy in enumerate(getattr(dataset, dataset_part)):
#     #         # print('#', i, eval_dataset.batch_size,eval_dataset.test_len)
#     #         if dataset._batch_size:
#     #             if (i+1)*dataset._batch_size>dataset._test_len:
#     #                 break
#     #             x.append(xy[0].numpy())
#     #             y.append(xy[1].numpy())
#     #         else:
#     #             #print(xy[elem_idx], xy[elem_idx].shape)
#     #             x.append(xy[0].numpy().reshape(xy[0].shape[0],1).T)
#     #             y.append(xy[1].numpy().reshape(xy[1].shape[0],1).T)
                
#     #     samples_x = np.concatenate(x, axis=0)
#     #     samples_y = np.concatenate(y, axis=0)
#     #     # samples = np.concatenate(l, axis=1).T
             
#     # else:
#     #     samples_x= getattr(getattr(dataset, dataset_part),'features')
#     #     samples_y= getattr(getattr(dataset, dataset_part),'labels')

#     if not idx_samples:
#             idx_samples= np.random.randint(len(samples_x))

#     if idx_samples=='all':
#         return samples_x, samples_y

#     if isinstance(idx_samples, int):
#         idx_samples= [idx_samples]

#     if isinstance(idx_samples, list):
#             samples_x= samples_x[idx_samples]  
#             samples_y= samples_y[idx_samples]  
              
#     return samples_x, samples_y




# def load_eidors_solution(initialdir:str=''):
#     initialdir= initialdir or os.getcwd()

#     # perm_eidors_path= os.path.join(os.path.split(metadata.idx_samples_file[0])[0], 'elems_solved.mat')
#     tmp= MatlabSamples()
#     pred_eidors=[]
#     try: 
#         for _ in range(5):
#             pred, file_path=load_predictions_EIDORS(initialdir=initialdir)
#             filename= os.path.splitext(os.path.split(file_path)[1])[0]
#             pred_eidors.append([scale_prepocess(pred['elem_data'].T, True), filename])
#     except LoadCancelledException as e :
#         logger.info(f'Loading eidors cancelled : ({e})')

#     return pred_eidors





if __name__ == "__main__":

    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    a= []

    l=[]
    l.append(None)
    l.append(a)
    l.append(a)
    print(l, l ==[])

    path= "E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/DStest/test10_infos2py.mat" 

    print(os.path.split(os.path.split(path)[0]))
    # 
    # raw_data= get_XY_from_MalabDataSet(path= path, data_sel= ['Xh','Yh'])
    
    # training_dataset=dataloader(raw_data, verbose=True, batch_size=1)
    # for inputs, indexes in training_dataset.train.as_numpy_iterator():
    #         print(inputs,'indexes', indexes)
    #         # Print the first element and the label
    #         print(inputs[0])
    #         print('label of this input is', inputs[1])
    #         break
    # pass

