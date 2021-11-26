


from eit_ai.train_utils.gen import Generators
from eit_ai.train_utils.lists import ListGenerators
# from eit_ai.keras.gen import GeneratorKeras
# from eit_ai.pytorch.gen import GeneratorPyTorch
from eit_ai.train_utils.metadata import MetaData


class WrongGeneratorError(Exception):
    """"""

def select_gen(metadata:MetaData) -> Generators:

    try:
        return GEN_LIST[ListGenerators(metadata.gen_type)]()
    except ValueError:
        raise WrongGeneratorError(f'Wrong generator type: {metadata.gen_type}')

def select_keras():
    from eit_ai.keras.gen import GeneratorKeras
    return GeneratorKeras()

def select_pytorch():
    from eit_ai.pytorch.gen import GeneratorPyTorch
    return GeneratorPyTorch()


GEN_LIST={
    ListGenerators.Keras : select_keras,
    ListGenerators.Pytorch: select_pytorch
}





if __name__ == "__main__":
    from eit_ai.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    print(ListGenerators.Keras.value)
    print(ListGenerators())




