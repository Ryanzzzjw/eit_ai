# list all models and dataset present in the package





from enum import Enum



class ListGenerators(Enum):
    Keras='Keras'
    Pytorch='Pytorch'


class ListModels(Enum):
    """"""

class KerasModels(ListModels):
    StdKerasModel='StdKerasModel'
    StdAutokerasModel='StdAutokerasModel'

class PytorchModels(ListModels):
    StdTorchModel='StdPytorchModel',

class ListDatasets(Enum):
    """"""

class KerasDatasets(ListDatasets):
    StdDataset='StdDataset'
    TfDataset='TfDataset'

class PytorchDatasets(ListDatasets):
    StdDataset='StdPytorchDataset'


class ListOptimizers(Enum):
    """ """
    
class ListLosses(Enum):
    """ """


if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)