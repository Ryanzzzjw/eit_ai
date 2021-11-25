

import os
import sys
import traceback
from logging import error, getLogger

import numpy as np
from eit_tf_workspace.raw_data.raw_samples import RawSamples
from glob_utils.files.files import (FileExt, OpenDialogFileCancelledException,
                                    WrongFileExtError, NotFileError, check_file,
                                    dialog_get_file_with_ext, load_mat)

logger = getLogger(__name__)

################################################################################
# Matlab Samples 
################################################################################

MATLAB_DATASET_VAR_KEYS=['X','y']



class MatlabSamples(RawSamples):

    def load(
        self,
        file_path:str=None,
        nb_samples2load:int=0,
        data_sel:list[str]=['Xih','Yih'],
        exit:bool= True)->None:
        """Errors handling of the loading process "_load"

        If an Error or an Exception occurs during the loading process
        the whole system will exit if the exit flag is set to `True`
        
        Args:
            file_path (str, optional): see "_load". Defaults to `None`.
            nb_samples2load (int, optional): see "_load". Defaults to `0`.
            data_sel (list[str], optional): see "_load". 
            Defaults to `['Xih','Yih']`.
            exit (bool, optional): exit flag. Defaults to `True`.
        """        
        er=None
        try:
            self._load(
                file_path=file_path, 
                nb_samples=nb_samples2load, 
                data_sel=data_sel)
        except (
            OpenDialogFileCancelledException, 
            WrongFileExtError,
            NotFileError) as e:
            logger.critical(f'Loading aborted: {e}')
            er=1
        except BaseException as e:
            er=1
            traceback.print_exc()
            logger.critical(f'Loading Cancelled: {e}')

        if exit and er:
            sys.exit()

    def _load(self, file_path:str, nb_samples:int, data_sel:list[str])->None:
        """Loading process of Matlab Samples

        Args:
            file_path (str, optional): Path of a matlab eit_dataset matfile.
            if not passed or is wrong the user will be ask to select one.
            Defaults to `None` (user will be ask).
            nb_samples2load (int, optional): number of samples to 
            load out of the matlab dataset, if 0 user will be asked on the 
            terminal. Defaults to `0`.
            data_sel (list[str], optional): Keys of the loaded data. 
            see "_XY_selection".Defaults to `['Xih','Yih']`.

        """      
        self.loaded=False
        self._extract_metadata_from_dataset_matfile(file_path)
        self._load_samples(nb_samples=nb_samples)
        self._XY_selection(data_sel=data_sel)
        self.loaded=True
    
    def _extract_metadata_from_dataset_matfile(self, file_path:str)->None:
        """Extract/Sort the data contained in the *info2py.mat-file from matlab
        dataset.

        this method set:
        - self.file_path : file path of the *info2py.mat-file from matlab
        dataset
        - self.dir_path : directory of this file/ of the dataset

        - self.dataset: metadata about the dataset (filenames, etc...)
        - self.user_entry: metadata about the user entries used for the 
        dataset generation in matlab
        - self.fwd_model: metadata about the eit fwd_model used for the 
        dataset generation in matlab (FEM, Stimulations, etc)

        Args:
            file_path (str): mat-file ending with *info2py.mat
        """
        #loading dataset mat-file
        var_dict, file_path= load_mat_file(
            file_path=file_path,
            title= 'Please select *infos2py.mat-files from a matlab dataset')
        self.file_path= file_path
        self.dir_path = os.path.split(file_path)[0]

        # Sorting of the variables in dataset, user_entry, fwd_model dicts 
        for key in var_dict:
            if ("userentry") in key:
                keynew= key.replace("userentry_", "")
                if ("fmdl") in key:
                    keynew= keynew.replace("fmdl_", "")   
                    self.fwd_model[keynew]= var_dict[key]
                else:
                    self.user_entry[keynew]= var_dict[key]
            else:
                self.dataset[key]= var_dict[key]

        # Samples folder /filenames extract
        self.dataset["samplesfolder"]= str_cellarray2str_list(
            self.dataset["samplesfolder"])
        self.dataset["samplesfilenames"]= str_cellarray2str_list(
            self.dataset["samplesfilenames"])
        self.dataset["samplesindx"]= self.dataset["samplesindx"]
        
        # Matlab used a one indexing system
        self.fwd_model['elems']= self.fwd_model['elems']-int(1) 

        logger.debug(f'Keys of dataset: {list(self.dataset.keys())}')
        logger.debug(f'Keys of fwd_model:{list(self.fwd_model.keys())}')
        logger.debug(f'Keys of user_entry:{list(self.user_entry.keys())}')

    def _load_samples(
        self, 
        nb_samples:int=0, 
        var_keys=MATLAB_DATASET_VAR_KEYS)->None:
        """Load the samples from each batch samples mat-files

        this method set:
        - self.samples: a dict with keys 'var_keys'

        Args:
            nb_samples (int, optional): [description]. Defaults to 0.
            var_keys ([type], optional): variables keys loaded from the batch
            samples mat-files. Defaults to MATLAB_DATASET_VAR_KEYS.

        """    
        # get the folder and filename of all batch samples mat-files 
        folder=os.path.join(self.dir_path, self.dataset["samplesfolder"][0])
        samples_batch_files= self.dataset["samplesfilenames"]
        samples_batch_paths= [
            os.path.join(folder,file) for file in samples_batch_files]
        # 
        self._set_nb_samples(nb_samples)
        idx_batch_file, idx_last_samples = self._idx_batch_loading()
        self._check_keys_in_batch_sample_files(samples_batch_paths,var_keys)

        for idx_batch in range(idx_batch_file+1):
            batch_file_path= samples_batch_paths[idx_batch]
            logger.info(
                f'Loading batch samples file : ...{batch_file_path[-50:]}')
            batch_file=load_mat(batch_file_path, logging=False)
            for key in self.samples.keys():
                if idx_batch==0:
                    if idx_batch==idx_batch_file:
                        s= [slice(None)]*batch_file[key].ndim
                        s[1]= slice(0,idx_last_samples+1)                    
                        self.samples[key]=batch_file[key][tuple(s)]
                    else:
                        self.samples[key]=batch_file[key]
                elif idx_batch==idx_batch_file:
                    s= [slice(None)]*batch_file[key].ndim
                    s[1]= slice(0,idx_last_samples+1)                    
                    self.samples[key]=np.append(
                        self.samples[key],batch_file[key][tuple(s)],axis=1)
                else:
                   self.samples[key]=np.append(
                       self.samples[key],batch_file[key],axis=1)
        # Logging
        for key in self.samples.keys():
            logger.debug(f'Size of "{key}": {self.samples[key].shape}')
           
    def _XY_selection(self, data_sel:list[str]= ['Xih','Yih'])->None:
        """ 

        this method set `self.X` (nb_samples, nbfeatures) and 
        `self.Y` (nb_samples, nblabels)

        Args:
            data_sel (list, optional): [description]. Defaults to ['Xih','Yih'].
        
        Note:
        - X and Y have shape (nb_samples, nbfeatures) and (nb_samples, nblabels)
        """
         # data selection
        tmpX = {
            #Voltages meas homogenious
            'Xh': self.samples['X'][:, :, 0],
            #Voltages meas inhomogenious
            'Xih': self.samples['X'][:, :, 1],
            #Voltages meas homogenious with noise  
            'Xhn': self.samples['X'][:, :, 2],
            #Voltages meas inhomogenious with noise  
            'Xihn': self.samples['X'][:, :, 3]
        }
        tmpY = {
            'Yh': self.samples['y'][:,:,0], # Image elem_data homogenious
            'Yih': self.samples['y'][:,:,1] # Image elem_data inhomogenious
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
            logger.warning(f'{data_sel=} are not availbles')
            data_sel= ['Xih','Yih']
    
        self.data_sel= data_sel        
        logger.debug(f'Data {data_sel} used')

        self.X= tmpX[data_sel[0]].T
        self.Y= tmpY[data_sel[1]].T
    
    def _set_nb_samples(self,nb_samples:int=0)->None:
        """Set nb of samples to load

        this method set self.nb_samples

        Args:
            nb_samples (int, optional): number of samples to load automatically.
             Defaults to `0` (user will be asked on terminal).
        """
        if not isinstance(nb_samples, int):
            nb_samples=0
        
        max_samples= np.amax(self.dataset["samplesindx"])
        self.nb_samples= max_samples # set the default number of samples

        if nb_samples < 0: # if negativ >> default value
            logger.warning(f'Number of samples to load negativ: {nb_samples=}')
        elif nb_samples==0: # if 0 >> ask user
            prompt= f"{max_samples} samples are availables. \n\
                Enter the number of samples to load (Enter for all): \n"
            input_user=input(prompt)
            try:
                self.nb_samples = int(input_user)
            except ValueError: # if Enter pressed or wrong entry >> default value
                logger.warning(
                    f'Nb of Samples should be an int: you enter {input_user}')
        elif nb_samples<=max_samples: # if lower than max value
            self.nb_samples=nb_samples
        else: # if higher than maxvalue >> default value
            logger.warning(
                f'Number of samples too high: {nb_samples=}>{max_samples=}')

        logger.info(f'{self.nb_samples} samples will be loaded')


    def _idx_batch_loading(self)-> tuple[int,int]:
        """Determine the indexes of the last batch samples file to load and
        in that one the last sample to load to obtain a total of 
        "self.nb_samples" samples
    
        Raises:
            Exception: if self.nb_samples is not before using "_set_nb_samples"

        Returns:
            tuple[int,int]: (index of last batch file, index of last sample)
        """
        if self.nb_samples == 0:
            raise Exception(
                'Please set nb_samples (_set_nb_samples) before')

        tmp= np.where(self.dataset["samplesindx"]==self.nb_samples)
        idx_last_batch, idx_last_samples= int(tmp[0][0]),int(tmp[1][0])

        return idx_last_batch, idx_last_samples

    def _check_keys_in_batch_sample_files(
        self, 
        batch_file_paths:list[str], 
        keys:list[str])->None:
        """ Check if the passed keys are available in the batch samples files
        (only the first will be checked)
        
        this method initialize the dict self.samples with the given keys
        and val is np.array([])

        Args:
            batch_file_paths (list[str]): batch samples files paths
            keys (list[str]): variables keys to load
        """ 

        batch_file=load_mat(batch_file_paths[0],logging=False)
        keys_batch_file= list(batch_file.keys())
        # check if each keys is in the keys_batch_file available
        keys_available=True
        for key in keys:
            if key not in keys_batch_file:
                logger.warning(
                    f'Variable-{key=} not available in batch samples files')
                keys_available=False

        if not keys_available: #keys_batch_file != keys: 
            keys=keys_batch_file

        logger.debug(
            f'Variables: {keys} will be loaded from the batch samples files')
        for key in keys:
            self.samples[key]= np.array([])


################################################################################
# Conversion methods from matlab to python
################################################################################


def str_cellarray2str_list(str_cellarray):
    """ After using loadmat, the str cell array have a strange shape
        >>> here the loaded "strange" array is converted to an str list

    Args:
        str_cellarray ( ndarray): correponing to str cell array in matlab

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
      
def load_mat_file(file_path:str=None,**kwargs)-> tuple[dict, str]:
    """Load a matlab mat-file passed as argument. If not passed the user will 
    be askto select a matlab mat-file.
    All variables contained in a mat-file (except the private var) are then 
    return in a dictionnary, also the file_path wil be returned
    (in case of selection by user)

    Args:
        file_path (str, optional): matlab mat-file to load. Defaults to None.
        **kwargs: translit to dialog_get_file_with_ext

    Returns:
        tuple[dict, str]: variables dict and file path
    """    

    if check_file(file_path,ext=FileExt.mat) is None:
        file_path= dialog_get_file_with_ext(ext=FileExt.mat, **kwargs)
    var_dict= load_mat(file_path)
    return var_dict, file_path
                                                                                
if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    # load_mat_file()
    file_path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    r= MatlabSamples()
    r.load()
