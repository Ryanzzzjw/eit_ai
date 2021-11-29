import os
import sys
from dataclasses import dataclass
from logging import error, getLogger

from eit_tf_workspace.train_utils.lists import (ListDatasets, ListGenerators,
                                                ListLosses, ListModels,
                                                ListOptimizers)
from glob_utils.files.files import (read_txt, save_as_mat, save_as_pickle,
                                    save_as_txt)
from glob_utils.log.msg_trans import highlight_msg
from glob_utils.pth.inout_dir import DEFAULT_DIRS
from glob_utils.pth.path_utils import (OpenDialogDirCancelledException,
                                       get_date_time, get_dir, get_POSIX_path,
                                       mk_new_dir)

logger = getLogger(__name__)

METADATA_FILENAME= 'metadata'
IDX_FILENAME= 'idx_samples'

################################################################################
# Class MetaData
################################################################################
@dataclass
class MetaData(object):
    """ Metadata Class regroup the data and information of the training
    for the training and eval"""
    time:str=None
    training_name:str=None
    dir_path:str=None

    raw_src_file:list[str]=None
    # dataset_src_file_pkl:list[str]=None
    idx_samples_file:list[str]=None
    model_saving_path:list[str]=None
    save_summary:bool=None

    data_select:list[str]=None
    _nb_samples:int=None
    batch_size:int=None
    test_ratio:float=None
    val_ratio:float=None
    # use_tf_dataset:bool=None
    normalize:list[bool]=None
    idx_samples:dict= None
    epoch:int=None
    max_trials_autokeras:int=None
    _train_len:int=None
    _val_len:int=None
    _test_len:int=None
    input_size:int=None
    output_size:int=None
    _steps_per_epoch:int =None
    _validation_steps:int =None
    _test_steps:int=None
    callbacks=None
    optimizer:str=None
    learning_rate:float= None
    loss:str= None
    metrics:list[str]= None

    training_duration:str=None
    gen_type:ListGenerators=None
    model_type:ListModels=None
    dataset_type:ListDatasets=None

    def __post_init__(self):
        self.set_idx_samples(save=False)
    
    def set_ouput_dir(self, training_name:str='', append_date_time:bool= True) -> None:
        """Create the ouput directory for training results

        Args:
            training_name (str, optional): if empty training_name='training_default_name'. Defaults to ''.
            append_date_time (bool, optional): Defaults to True.
        """

        self.time = get_date_time()
        if not training_name:
            training_name='training_default_name'
        self.training_name= f'{training_name}_{self.time}' if append_date_time else training_name
        self.dir_path= mk_new_dir(
            self.training_name,
            parent_dir=DEFAULT_DIRS.output)
        msg=f'Training results will be found in : {self.dir_path}'
        logger.info(highlight_msg(msg))

    def set_model_dataset_type(self, gen_type:ListGenerators, model_type:ListModels, dataset_type:ListDatasets):
        """"""
        self.gen_type= gen_type.value
        self.model_type= model_type.value
        self.dataset_type= dataset_type.value

    def set_4_dataset(  
            self, 
            batch_size:int=32,
            test_ratio:float=0.2,
            val_ratio:float=0.2, 
            # use_tf_dataset:bool=False, 
            normalize= [True, True]):
        """ """             
        self.batch_size = batch_size
        self.val_ratio, self.test_ratio =check_ratios(val_ratio, test_ratio)
        self.normalize=normalize 

    def set_4_model(   
            self,
            epoch:int=10,
            max_trials_autokeras=10, 
            callbacks=[],
            optimizer:ListOptimizers=None,
            learning_rate:float=None,
            loss:ListLosses=None,
            metrics=['mse'],
            save_summary:bool=False)-> None:
        """ """
        if not self.batch_size:
            error('call first set_4_dataset')

        self.epoch= epoch
        self.max_trials_autokeras=max_trials_autokeras
        self.callbacks=callbacks      
        self.optimizer=optimizer.value if optimizer else None
        self.learning_rate= learning_rate
        self.loss=loss.value if loss else None
        self.metrics=metrics
        self.save_summary=save_summary

        self._steps_per_epoch=compute_steps(self.batch_size, self._train_len)
        self._validation_steps=compute_steps(self.batch_size, self._val_len)
        self._test_steps=compute_steps(self.batch_size, self._test_len)

    def set_raw_src_file(self, src_file):
        self.raw_src_file=make_PoSIX_abs_rel(src_file, self.dir_path)
        
    def set_model_saving_path(self, model_saving_path):
        self.model_saving_path=make_PoSIX_abs_rel(model_saving_path, self.dir_path)

    def set_idx_samples(self, idx_train:list=[], idx_val:list=[], idx_test:list=[], save:bool=True):
        self.idx_samples={
            'idx_train': idx_train,
            'idx_val': idx_val,
            'idx_test': idx_test
        }
        if save:
            self.save_idx_samples()

    def get_idx_samples(self):
        return [
            self.idx_samples['idx_train'],
            self.idx_samples['idx_val'],
            self.idx_samples['idx_test'],
        ] 

    def set_idx_samples_file(self, path):
        self.idx_samples_file=make_PoSIX_abs_rel(path, self.dir_path)

    def save_idx_samples(self):
        """ save the indexes of the samples used to build 
        the dataset train, val and test """

        indexes = self.idx_samples
        time = self.time or get_date_time()
        path =  os.path.join(self.dir_path, f'{IDX_FILENAME}_{time}')
        save_as_mat(path, indexes)
        save_as_pickle(path, indexes)
        save_as_txt(path,indexes)
        self.set_idx_samples_file(path)

    def set_training_duration(self, duration:str=''):
        self.training_duration= duration

    def set_4_raw_samples(self, data_sel):
        self.data_select=data_sel

    def save(self, dir_path= None):

        if not self.dir_path:
            return
        dir_path = dir_path or self.dir_path
        filename=os.path.join(dir_path,METADATA_FILENAME)
        copy=MetaData()
        for key, val in self.__dict__.items():
            if hasattr(val, '__dict__'):
                setattr(copy, key, type(val).__name__)
            elif isinstance(val, list):
                l = []
                for elem in val:
                    if hasattr(elem, '__dict__'):
                        l.append(type(elem).__name__)
                    else:
                        l.append(elem)
                setattr(copy, key, l)
            else:
                setattr(copy, key, val)
        save_as_txt(filename, copy)
        logger.info(highlight_msg(f'Metadata saved in: {filename}'))
        
        
    def read(self, path):
        
        load_dict=read_txt(path)
        for key in load_dict.keys():
            if key in self.__dict__.keys():
                setattr(self,key, load_dict[key])

        logger.info(highlight_msg(f'Metadata loaded from: {path}, '))
        logger.info(f'Metadata loaded :\n{self.__dict__.keys()}')
        logger.debug(f'Metadata loaded (details):\n{self.__dict__}')
        self.dir_path=os.path.split(path)[0]

    def reload(self, dir_path:str=''):

        if not os.path.isdir(dir_path):
            title= 'Select directory of model to evaluate'
            try: 
                dir_path=get_dir(title=title)
            except OpenDialogDirCancelledException as e:
                logger.critical('User cancelled the loading')
                sys.exit()
        try:   
            self.read(os.path.join(dir_path,METADATA_FILENAME))
        except FileNotFoundError as e:
            logger.critical(f'File "{METADATA_FILENAME}" not found in folder:\n{dir_path}\n({e})')
            sys.exit()


################################################################################
# Methods
################################################################################

def compute_steps(batch_size:int, len_dataset:int)->int :
    return len_dataset // batch_size if batch_size or len_dataset==0 else None

def check_ratios(val_ratio:float, test_ratio:float)-> tuple[float, float]:
    """Check the ratios of val and test dataset"""
    if val_ratio <=0.0:
        val_ratio=0.2
        logger.warning(f'val ratio <=0.0: set to {val_ratio}')

    if test_ratio <=0.0:
        test_ratio=0.2
        logger.warning(f'test ratio <=0.0: set to {test_ratio}')

    if test_ratio+val_ratio>=0.5:
        test_ratio=0.2
        val_ratio=0.2
        logger.warning(f'val and test ratios:{val_ratio} and {test_ratio}')
    return val_ratio, test_ratio

def make_PoSIX_abs_rel(path:str, rel_path:str)-> list[str]:
    rel=os.path.relpath(path, start=rel_path)
    return [ get_POSIX_path(path), get_POSIX_path(rel)]

################################################################################
# Methods for use  metadata
################################################################################


def reload_metadata(dir_path:str='')-> MetaData:
    """"""
    metadata=MetaData()
    metadata.reload(dir_path)
    return metadata


if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    a= MetaData()
    a.reload()
    


