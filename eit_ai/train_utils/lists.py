# list all models and dataset present in the package





from enum import Enum
from typing import Any, Union



"""Those Enumerations lists are used save and select classes

"""

class ExtendedEnum(Enum):

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_keys_name(cls):
        return cls._member_names_

    @classmethod
    def list_items(cls):
        return cls._member_map_


################################################################################
# Generators
################################################################################
class ListWorkspaces(ExtendedEnum):
    Keras='Keras'
    PyTorch='PyTorch'

################################################################################
# Model Handlers
################################################################################

class ListModelHandlers(ExtendedEnum):
    """"""
class ListKerasModelHandlers(ListModelHandlers):
    KerasModelHandler='KerasModelHandler'
    AutokerasModelHandler='AutokerasModelHandler'
class ListPytorchModelHandlers(ListModelHandlers):
    PytorchModelHandler='PytorchModelHandler'

################################################################################
# Models
################################################################################
class ListModels(ExtendedEnum):
    """"""
class ListKerasModels(ListModels):
    StdKerasModel='StdKerasModel'
    StdAutokerasModel='StdAutokerasModel'
class ListPytorchModels(ListModels):
    StdPytorchModel='StdPytorchModel'
    Conv1dNet='Conv1dNet'
    AutoEncoder='AutoEncoder'

# ################################################################################
# # Dataset 
# ################################################################################

# class ListDatasets(ExtendedEnum):
#     """"""
# class ListKerasDatasets(ListDatasets):
#     StdDataset='StdDataset'
#     TfDataset='TfDataset'
# class ListPytorchDatasets(ListDatasets):
#     StdPytorchDataset='StdPytorchDataset'

################################################################################
# Datasets
################################################################################

class ListDatasetHandlers(ExtendedEnum):
    """"""
class ListKerasDatasetHandlers(ListDatasetHandlers):
    KerasDatasetHandler='StdDatasetHandler'
    TfDatasetHandler='TfDatasetHandler'
class ListPytorchDatasetHandlers(ListDatasetHandlers):
    StdPytorchDatasetHandler='StdPytorchDatasetHandler'
    PytorchConv1dDatasetHandler='PytorchConv1dDatasetHandler'
    # PytorchUxyzDatasetHandler= 'PytorchUxyzDatasetHandler'

################################################################################
# Optimizers
################################################################################

class ListOptimizers(ExtendedEnum):
    """"""
class ListKerasOptimizers(ListOptimizers):
    Adam='Adam'
class ListPyTorchOptimizers(ListOptimizers):
    Adam='Adam'
    SGD='SGD'

################################################################################
# Losses
################################################################################

class ListLosses(ExtendedEnum):
    """"""
class ListKerasLosses(ListLosses):
    CategoricalCrossentropy='CategoricalCrossentropy'
    MeanSquaredError='MeanSquaredError'

class ListPyTorchLosses(ListLosses):
    MSELoss='MSELoss'
    CrossEntropyLoss='CrossEntropyLoss'


class ListNormalizations(ExtendedEnum):
    Identity='Identity'
    MinMax_01='MinMax01'
    MinMax_11='MinMax-11'
    Norm='Norm'




################################################################################
# Methods
################################################################################


def get_from_dict(
    list_item:Union[str, Enum],
    dict_obj:dict, 
    list_instance:Enum,
    return_listobj:bool=False)->tuple[Any, Enum]:
    """[summary]

    Args:
        list_item (Union[str, Enum]): [description]
        dict_obj (dict): [description]
        list_instance (Enum): [description]

    Raises:
        ValueError: [description]

    Returns:
        tuple[Any, Enum]: [description]
    """    
    if isinstance(list_item, Enum):
        list_item=list_item.value
    try:


        if return_listobj:
            return dict_obj[list_instance(list_item)], list_instance(list_item)
        return dict_obj[list_instance(list_item)]
    except ValueError:
        raise ValueError(f'Wrong {list_instance.__name__}: {list_item}')


if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    from glob_utils.debug.debugging_help import print_obj_type_dict


    class TestEnum(ExtendedEnum):
        name1='val1',
        name2= 2
    TEST={
        TestEnum.name1:'test1',
        TestEnum.name2:'test2'
    }
    print_obj_type_dict(TEST.keys())
    print(isinstance(ListWorkspaces.Keras, TestEnum))
    print(list(ListPyTorchOptimizers))


    