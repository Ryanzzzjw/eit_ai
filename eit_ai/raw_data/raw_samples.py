from abc import ABC, abstractmethod
import logging

from eit_ai.train_utils.metadata import MetaData


logger = logging.getLogger(__name__)


################################################################################
# Abstract Class of Raw Samples
################################################################################

class RawSamples(ABC):
    """Abstract Class to store external generated dataset
    """    
    
    def __init__(self) -> None:
        super().__init__()
        self.dataset = {}
        self.fwd_model= {}
        self.user_entry = {}
        self.setup = {}
        self.sim = {}
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
        file_path:str=None,
        nb_samples:int=0,
        data_sel=None,
        time:str= None):
        """Loading process of samples

        should set:
        
        self.dataset
        self.fwd_model
        self.user_entry
        self.sim
        self.setup
        self.samples
        self.X: ArrayLike(n_samples, n_features)
        self.Y: ArrayLike(n_samples, n_labels)
        self.file_path
        self.dir_path
        self.loaded
        self.nb_samples

        Args:
            file_path (str, optional): Path of the external dataset .
            if not passed or is wrong the user will be ask to select one.
            Defaults to `None` (user will be ask).
            nb_samples (int, optional): number of samples to 
            load out of the external dataset, if 0 user will be asked on the 
            terminal. Defaults to `0`.
            data_sel (list[str], optional): Filterin /selection of the specific
            loaded data. Defaults to `['Xih','Yih']`.

        """ 


################################################################################
# Methods using Rawdata
################################################################################

def load_samples(raw_samples:RawSamples, src_path:str, metadata:MetaData)-> RawSamples:
    """"""
    raw_samples=raw_samples
    raw_samples.load(
        file_path=src_path,
        nb_samples=metadata._nb_samples,
        data_sel= metadata.data_select)
    metadata.set_raw_src_file(raw_samples.file_path)
    return raw_samples

def reload_samples(raw_samples:RawSamples, metadata:MetaData)->RawSamples:
    """"""
    raw_samples.load(
        file_path=metadata.raw_src_file[0],
        nb_samples=metadata._nb_samples,
        data_sel= metadata.data_select)
    return raw_samples



if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    reload_samples(1)
    """"""