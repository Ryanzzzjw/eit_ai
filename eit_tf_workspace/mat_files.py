

import os
from logging import error
from typing import List


import numpy as np

from scipy.io import loadmat

from eit_tf_workspace.path_utils import DialogCancelledException, verify_file,get_file, load_pickle, get_date_time, get_POSIX_path, save_as_pickle

import eit_tf_workspace.constants as const
from eit_tf_workspace.utils.log import MAX_LOG_MSG_LENGTH, main_log, log_file_loaded

from logging import getLogger, error
logger = getLogger(__name__)



# Conversion methods from matlab to python

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

# loading of mat-file

class LoadCancelledException(Exception):
    """"""
class WrongFileTypeSelectedError(Exception):
    """"""

def get_file_dir_path( file_path:str=""):
    file_path= verify_file(file_path, extension=const.EXT_MAT)
    if not file_path:
        try: 
            file_path =get_file(
                title= f'Please select *{const.EXT_MAT} files',
                filetypes=[("Matlab file",f"*{const.EXT_MAT}")],
                split=False)
        except DialogCancelledException:
            raise LoadCancelledException('Loading aborted from user')
    dir_path=os.path.split(file_path)[0]

    if not verify_file(file_path, extension=const.EXT_MAT):
        raise WrongFileTypeSelectedError('User selected wrong file!')

    return dir_path , file_path

def load_predictions_EIDORS(file_path):
    """     """
    _ , file_path= get_file_dir_path(file_path)
    file = loadmat(file_path)
    samples_EIDORS = {key: file[key] for key in file.keys() if ("__") not in key}
    logger.debug(f'Loaded keys: {samples_EIDORS.keys()} from file {os.path.split(file_path)[1]}')
    return samples_EIDORS

class MatlabDataSet(object):
    def __init__(self) -> None:
        super().__init__()
        self.type= 'MatlabDataSet'
        self.dataset = {}
        self.fwd_model= {}
        self.user_entry = {}
        self.samples = {}
        # self.samples_EIDORS = {}
        # self.path_pkl= ''
        self.X= []
        self.Y=[]
        self.file_path= ''
        self.dir_path= ''

    # def flex_load(self, path="", auto=False, type2load=const.EXT_MAT, time=None):

    #     if verify_file(path, extension=const.EXT_MAT):
    #         self.load_dataset_from_mat_file(file_path=path, auto= auto, time=time)
    #     elif verify_file(path, extension=const.EXT_PKL):
    #         self.load_dataset_from_pickle(path)
    #     elif type2load==const.EXT_MAT:
    #         self.load_dataset_from_mat_file(auto=auto, time=time)
    #     else:
    #         self.load_dataset_from_pickle(path)

    def load_dataset_from_mat_file(self, file_path:str="", nb_samples2load:int=0, data_sel=['Xih','Yih'], time:str= None):
        """      """
        self.dir_path , self.file_path= get_file_dir_path(file_path)
        log_file_loaded(file_path=self.file_path)
        self._load_metadata_from_dataset_matfile(self.file_path)
        self._load_samples(nb_samples2load=nb_samples2load)
        self._XY_selection(data_sel=data_sel)
        # self.save_dataset(time=time)

    def _XY_selection(self, data_sel= ['Xih','Yih']):
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

        self.X= tmpX[data_sel[0]]
        self.Y= tmpY[data_sel[1]]

    # def load_dataset_from_pickle(self, path=""):
    #     """load a MatlabDataSet from a pickle-file

    #     Returns:
    #         loaded_dataset[MatlabDataSet]: obvious
    #     """

    #     if verify_file(path, extension=const.EXT_PKL):
    #         self.path, self.file_path= os.path.split(path)
    #     else:
    #         self.path, self.file_path= get_file(filetypes=[("pickle file","*.pkl")])

    #     path= self.path
    #     file_path= os.path.join(self.path, self.file_path)
    #     if verify_file(file_path, extension=const.EXT_PKL):
            
    #         self= load_pickle(file_path, class2upload=self)
    #         self.path_pkl=file_path # as we do not save the pickel we have to actualizate the path (win/unix)
    #         self.path= path # we have to actualizate the path (win/unix)
    #         self.load_samples(mode='reload')
    #         #self.save_dataset()
    #     else:
    #         pass

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
                if idx_batch==idx_lastbatch2load:
                    s= [slice(None)]*batch_file[key].ndim
                    s[1]= slice(0,idx_lastsamplesoflastbatch2load+1)                    
                    self.samples[key]=np.append(self.samples[key],batch_file[key][tuple(s)],axis=1)
                elif idx_batch==0:
                    self.samples[key]=batch_file[key]
                else:
                   self.samples[key]=np.append(self.samples[key],batch_file[key],axis=1)
        for key in self.samples.keys():
            logger.debug(f'Size of sample loaded "{key}": {self.samples[key].shape}')

    # def __save_dataset(self, time= None):
    #     """ save the MatlabDataSet under a pickle-file
    #             (samples are cleared to avoid a big file)
    #     """
    #     time = time or get_date_time()
    #     filename= os.path.join(self.dir_path, f'{time}{const.EXT_PKL}')
    #     tmp= self.samples
    #     self.samples = {}
    #     save_as_pickle(filename,self)
    #     self.path_pkl= get_POSIX_path(filename)
    #     self.samples = tmp


def get_MalabDataSet(file_path="", data_sel= ['Xih','Yih'], nb_samples2load:int=0) -> MatlabDataSet:
    """[summary]

    Args:
        path (str, optional): [description]. Defaults to "".
        data_sel (list, optional): [description]. Defaults to ['Xih','Yih'].

    Returns:
        [type]: [description]
    """
    raw_data=MatlabDataSet()
    try:
        raw_data.load_dataset_from_mat_file(file_path=file_path,nb_samples2load=nb_samples2load, data_sel=data_sel)        
    except (LoadCancelledException, WrongFileTypeSelectedError) as e:
         logger.warning(f'Loading Cancelled: {e}')
    except BaseException as e:
        logger.error(e)
    finally:
        return raw_data

                                                                                
if __name__ == "__main__":
    main_log()
    file_path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    get_MalabDataSet(file_path=file_path, nb_samples2load=10000)