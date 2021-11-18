from abc import ABC, abstractmethod
from logging import getLogger

from eit_tf_workspace.train_utils.metadata import MetaData


logger = getLogger(__name__)


################################################################################
# Abstract Class of Raw Samples
################################################################################

class RawSamples(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self.dataset = {}
        self.fwd_model= {}
        self.user_entry = {}
        self.samples = {}
        self.X= []
        self.Y=[]
        self.file_path= ''
        self.dir_path= ''
        self.loaded:bool=False
        self.nb_samples:int=0

    def is_loaded(self)-> bool:
        return self.loaded

    @abstractmethod
    def load(
        self,
        file_path:str='',
        nb_samples2load:int=0,
        data_sel=None,
        time:str= None):
        """"""


################################################################################
# Methods using Rawdata
################################################################################

def load_samples(raw_samples:RawSamples, src_path:str, metadata:MetaData)-> RawSamples:
    """"""
    raw_samples=raw_samples
    raw_samples.load(
        file_path=src_path,
        nb_samples2load=metadata._nb_samples,
        data_sel= metadata.data_select)
    metadata.set_raw_src_file(raw_samples.file_path)
    return raw_samples

def reload_samples(raw_samples:RawSamples, metadata:MetaData)->RawSamples:
    """"""
    raw_samples.load(
        file_path=metadata.raw_src_file[0],
        nb_samples2load=metadata._nb_samples,
        data_sel= metadata.data_select)
    return raw_samples



if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    reload_samples(1)
    """"""