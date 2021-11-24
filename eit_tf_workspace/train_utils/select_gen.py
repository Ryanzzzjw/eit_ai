


from eit_tf_workspace.train_utils.gen import Generators
from eit_tf_workspace.train_utils.lists import ListGenerators
from eit_tf_workspace.train_utils.metadata import MetaData

class WrongGeneratorError(Exception):
    """"""

def select_gen(metadata:MetaData) -> Generators:
    """Return a generator corresponding to the saved metadata.gen_type

    Args:
        metadata (MetaData)

    Raises:
        WrongGeneratorError: raised if gen_type is not listed in GEN_LIST

    Returns:
        [Generators]: Generator of Model and dataset

    """    
    try:
        return GEN_LIST[ListGenerators(metadata.gen_type)]()
    except ValueError:
        raise WrongGeneratorError(f'Wrong generator type: {metadata.gen_type}')

def select_gen_keras()-> Generators:
    """Return Keras generator

    Returns:
        [Generators]: Keras generator
    """

    from eit_tf_workspace.keras.gen import GeneratorKeras
    return GeneratorKeras()

def select_gen_pytorch()-> Generators:
    """Return Pytorch generator

    Returns:
        [Generators]: Pytorch generator
    """
    from eit_tf_workspace.pytorch.gen import GeneratorPyTorch
    return GeneratorPyTorch()


GEN_LIST={
    ListGenerators.Keras : select_gen_keras,
    ListGenerators.Pytorch: select_gen_pytorch
}


if __name__ == "__main__":
    from glob_utils.log.log  import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    print(ListGenerators.Keras.value)
    print(ListGenerators())




