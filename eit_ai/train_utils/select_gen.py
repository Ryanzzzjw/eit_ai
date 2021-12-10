


from eit_ai.train_utils.gen import Generators
from eit_ai.train_utils.lists import ListGenerators
from eit_ai.train_utils.metadata import MetaData

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

    from eit_ai.keras.gen import GeneratorKeras
    return GeneratorKeras()

def select_gen_pytorch()-> Generators:
    """Return Pytorch generator

    Returns:
        [Generators]: Pytorch generator
    """
    from eit_ai.pytorch.gen import GeneratorPyTorch
    return GeneratorPyTorch()


GEN_LIST={
    ListGenerators.Keras : select_gen_keras,
    ListGenerators.PyTorch: select_gen_pytorch
}


if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    print(ListGenerators.Keras.value)
    print(ListGenerators())




