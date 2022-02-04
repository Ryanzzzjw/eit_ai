


from eit_ai.train_utils.workspace import AiWorkspace
from eit_ai.train_utils.lists import ListWorkspaces
from eit_ai.train_utils.metadata import MetaData

class WrongGeneratorError(Exception):
    """"""

def select_workspace(metadata:MetaData) -> AiWorkspace:
    """Return a generator corresponding to the saved metadata.gen_type

    Args:
        metadata (MetaData)

    Raises:
        WrongGeneratorError: raised if gen_type is not listed in GEN_LIST

    Returns:
        [Generators]: Generator of Model and dataset

    """    
    try:
        return WORKSPACES[ListWorkspaces(metadata.workspace)]()
    except ValueError:
        raise WrongGeneratorError(f'Wrong generator type: {metadata.workspace}')

def select_gen_keras()-> AiWorkspace:
    """Return Keras generator

    Returns:
        [Generators]: Keras generator
    """

    from eit_ai.keras.workspace import KerasWorkspace
    return KerasWorkspace()

def select_gen_pytorch()-> AiWorkspace:
    """Return Pytorch generator

    Returns:
        [Generators]: Pytorch generator
    """
    from eit_ai.pytorch.workspace import PyTorchWorkspace
    return PyTorchWorkspace()


WORKSPACES={
    ListWorkspaces.Keras : select_gen_keras,
    ListWorkspaces.PyTorch: select_gen_pytorch
}


if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    print(ListWorkspaces.Keras.value)
    print(ListWorkspaces())




