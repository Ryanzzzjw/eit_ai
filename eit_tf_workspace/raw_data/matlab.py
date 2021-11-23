

import os
import sys
from logging import getLogger
import traceback
import numpy as np
from eit_tf_workspace.constants import EXT_MAT
from eit_tf_workspace.raw_data.raw_samples import RawSamples
from eit_tf_workspace.utils.log import log_file_loaded
from eit_tf_workspace.utils.path_utils import (LoadCancelledException,
                                               WrongFileTypeSelectedError,
                                               get_file_dir_path)
from scipy.io import loadmat

logger = getLogger(__name__)

################################################################################
# Matlab Samples 
################################################################################

class MatlabSamples(RawSamples):

    def load(self, file_path:str="", nb_samples2load:int=0, data_sel=['Xih','Yih'], time:str= None):
        try:
            self._load( file_path=file_path, nb_samples2load=nb_samples2load, data_sel=data_sel)
        except (LoadCancelledException, WrongFileTypeSelectedError) as e:
            logger.warning(f'Loading Cancelled: {e}')
            sys.exit()
        except BaseException as e:
            traceback.print_exc()
            logger.error(f'Loading Cancelled: {e}')
            sys.exit()

    def _load(self, file_path:str="", nb_samples2load:int=0, data_sel=['Xih','Yih'], time:str= None):
        self.loaded=False
        self.dir_path , self.file_path= get_file_dir_path(file_path, title= 'Please select *infos2py.mat-files from a matlab eit_dataset')
        log_file_loaded(file_path=self.file_path)
        self._load_metadata_from_dataset_matfile(self.file_path)
        self._load_samples(nb_samples2load=nb_samples2load)
        self._XY_selection(data_sel=data_sel)
        self.loaded=True
    
    def _XY_selection(self, data_sel= ['Xih','Yih']):
        """[summary]

        Args:
            data_sel (list, optional): [description]. Defaults to ['Xih','Yih'].
        
        Note:
        - X and Y have shape (nb_samples, nbfeatures) and (nb_samples, nblabels)
        """
         # data selection
        tmpX = {
            'Xh': self.samples['X'][:, :, 0],
            'Xih': self.samples['X'][:, :, 1],
            'Xhn': self.samples['X'][:, :, 2],
            'Xihn': self.samples['X'][:, :, 3],
        }
        tmpY = {
            'Yh': self.samples['y'][:,:,0],
            'Yih': self.samples['y'][:,:,1]
        }
        # here we can create the differences
        tmpX['Xih-Xh']= tmpX['Xih']-tmpX['Xh']
        tmpY['Yih-Yh']= tmpY['Yih']-tmpY['Yh']

        tmpX['Xihn-Xhn']= tmpX['Xihn']-tmpX['Xhn']
        tmpX['Xihn-Xh']= tmpX['Xihn']-tmpX['Xh']
        tmpX['Xihn-Xh']= tmpX['Xih']-tmpX['Xhn']
        ## control input

        if data_sel[0] not in tmpX.keys() or \
        data_sel[1] not in tmpY.keys():
            logger.warning('not correct data_sel')
            data_sel= ['Xih','Yih']
    
        self.data_sel= data_sel        
        logger.debug(f'Data {data_sel} used')

        self.X= tmpX[data_sel[0]].T
        self.Y= tmpY[data_sel[1]].T


    def _load_metadata_from_dataset_matfile(self, file_path):
        """ extract the data  contained in the *info2py.mat to load the samples in python.

        Args:
            filename (str): mat-file ending with *info2py.mat
            path (str): folder where the mat-file is to found
        """
        file = loadmat(file_path)

        for key in file.keys():
            if ("userentry") in key:
                keynew= key.replace("userentry_", "")
                if ("fmdl") in key:
                    keynew= keynew.replace("fmdl_", "")   
                    self.fwd_model[keynew]= file[key]
                else:
                    self.user_entry[keynew]= file[key]
            else:
                if ("__") not in key:
                    self.dataset[key]= file[key]
        # Samples folder /filenames extract
        self.dataset["samplesfolder"]= str_cellarray2str_list(self.dataset["samplesfolder"])
        self.dataset["samplesfilenames"]= str_cellarray2str_list(self.dataset["samplesfilenames"])
        self.dataset["samplesindx"]= self.dataset["samplesindx"]

        self.fwd_model['elems']= self.fwd_model['elems']-int(1)

        logger.debug(f'Keys of loaded mat file: {file.keys()}')
        logger.debug(f'Keys of dataset: {self.dataset.keys()}')
        logger.debug(f'Keys of fwd_model:{self.fwd_model.keys()}')
        logger.debug(f'Keys of user_entry:{self.user_entry.keys()}')
           
    
    def _set_nb_samples2load(self,nb_samples2load:int=0):
        """ Get nb of samples to load (console input from user)
        Args:
            number_samples2load (int, optional):if > 0 console input wont be asked. Defaults to 0.
        Returns:
            number_samples2load [int]: nb of samples to load
        """
        if not isinstance(nb_samples2load, int): #
            nb_samples2load=0
        
        max_samples= np.amax(self.dataset["samplesindx"])
        self.nb_samples= max_samples
        if nb_samples2load<0:
            logger.warning(f'Number of samples to load negativ: {nb_samples2load}')
        elif nb_samples2load==0:
            prompt= f"{max_samples} samples are availables. \nEnter the number of samples to load (Enter for all): \n"
            input_user=input(prompt)
            try:
                self.nb_samples = int(input_user)
            except ValueError:
                logger.warning(f'Nb of Samples should be an int: you enter {input_user}')
        elif nb_samples2load<=max_samples:
            self.nb_samples=nb_samples2load
        else:
            logger.warning(f'Number of samples to load too high: {nb_samples2load}>{max_samples}')
        logger.info(f'{self.nb_samples} samples will be loaded')


    def _get_idx_for_samples2loading(self,nb_samples2load:int=0):
        """ Get nb of samples to load (console input from user)
        Args:
            number_samples2load (int, optional):if > 0 console input wont be asked. Defaults to 0.
        Returns:
            number_samples2load [int]: nb of samples to load
        """
        self._set_nb_samples2load(nb_samples2load)

        tmp= np.where(self.dataset["samplesindx"]==self.nb_samples)
        idx_lastbatch2load= tmp[0][0]
        idx_lastsamplesoflastbatch2load= tmp[1][0]

        return idx_lastbatch2load, idx_lastsamplesoflastbatch2load

    def _verify_keys_of_samples(self, keys= ['X','y']):
        """ verify the keys present in the first batch_file 
        init the sample dict with given keys
        """
        keys2load= keys
        folder=os.path.join(self.dir_path, self.dataset["samplesfolder"][0])
        filesnames= self.dataset["samplesfilenames"]
        batch_file=loadmat(os.path.join(folder, filesnames[0]),)
        keys2load_frombatchfile= [ key for key in batch_file.keys() if "__" not in key]

        if keys2load_frombatchfile != keys2load:
            logger.warning(f'Samples file does not contain {keys2load} variables as expected')
            keys2load=keys2load_frombatchfile

        logger.debug(f'Variables of samples to load : {keys2load}')
        for key in keys2load:
            self.samples[key]= np.array([])

    def _load_samples(self, nb_samples2load:int=0):
        """ load the samples from each mat-file """
        folder=os.path.join(self.dir_path, self.dataset["samplesfolder"][0])
        filesnames= self.dataset["samplesfilenames"]

        idx_lastbatch2load, idx_lastsamplesoflastbatch2load=self._get_idx_for_samples2loading(nb_samples2load=nb_samples2load)
        self._verify_keys_of_samples(keys=['X', 'y'])
        
        for idx_batch in range(idx_lastbatch2load+1):
            batch_file_path= os.path.join(folder, filesnames[idx_batch])
            logger.info(f'Loading batch samples file : ...{batch_file_path[-50:]}')
            batch_file=loadmat(batch_file_path)
            for key in self.samples.keys():
                if idx_batch==0:
                    if idx_batch==idx_lastbatch2load:
                        s= [slice(None)]*batch_file[key].ndim
                        s[1]= slice(0,idx_lastsamplesoflastbatch2load+1)                    
                        self.samples[key]=batch_file[key][tuple(s)]
                    else:
                        self.samples[key]=batch_file[key]
                elif idx_batch==idx_lastbatch2load:
                    s= [slice(None)]*batch_file[key].ndim
                    s[1]= slice(0,idx_lastsamplesoflastbatch2load+1)                    
                    self.samples[key]=np.append(self.samples[key],batch_file[key][tuple(s)],axis=1)
                else:
                   self.samples[key]=np.append(self.samples[key],batch_file[key],axis=1)
        for key in self.samples.keys():
            logger.debug(f'Size of sample loaded "{key}": {self.samples[key].shape}')


# 'def get_matlab_dataset(file_path="", data_sel= ['Xih','Yih'], nb_samples2load:int=0) -> MatlabSamples:
#     """[summary]

#     Args:

#         path (str, optional): [description]. Defaults to "".
#         data_sel (list, optional): [description]. Defaults to ['Xih','Yih'].

#     Returns:
#         [type]: [description]
#     """
#     raw_samples=MatlabSamples()
#     try:
#         raw_samples.load(file_path=file_path,nb_samples2load=nb_samples2load, data_sel=data_sel)        
#         return raw_samples
#     except (LoadCancelledException, WrongFileTypeSelectedError) as e:
#         logger.warning(f'Loading Cancelled: {e}')
#         sys.exit()
#     except BaseException as e:
#         logger.error(e)
#         sys.exit()'

################################################################################
# Conversion methods from matlab to python
################################################################################


def str_cellarray2str_list(str_cellarray):
    """ After using loadmat, the str cell array have a strange shape
        >>> here the loaded "strange" array is converted to an str list

    Args:
        str_cellarray ("strange" ndarray): correponing to str cell array in matlab

    Returns:
        str list: 
    """
    if str_cellarray.ndim ==2: 
        tmp= str_cellarray[0,:]
        str_array= [ t[0] for t in tmp] 
    elif str_cellarray.ndim ==1:
        tmp= str_cellarray[0]
        str_array= [tmp] 
    return str_array 

################################################################################
# Loading of mat files
################################################################################
      
def load_mat_file(file_path:str='',**kwargs):
    """load all variables contained in a mat file in a dictionnary,
    return the dict and the file_path (in case of selection by user)"""
    _ , file_path= get_file_dir_path(file_path, extension=EXT_MAT, **kwargs)
    file = loadmat(file_path)
    var = {key: file[key] for key in file.keys() if ("__") not in key}
    logger.debug(f'Loaded keys: {var.keys()} from file {os.path.split(file_path)[1]}')
    return var, file_path
                                                                                
if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    file_path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    # get_matlab_dataset(file_path=file_path, nb_samples2load=10000)
