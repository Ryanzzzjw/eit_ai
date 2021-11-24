


from eit_tf_workspace.train_utils.gen import Generators
from eit_tf_workspace.train_utils.lists import ListGenerators
# from eit_tf_workspace.keras.gen import GeneratorKeras
# from eit_tf_workspace.pytorch.gen import GeneratorPyTorch
from eit_tf_workspace.train_utils.metadata import MetaData


class WrongGeneratorError(Exception):
    """"""

def select_gen(metadata:MetaData) -> Generators:

    try:
        return GEN_LIST[ListGenerators(metadata.gen_type)]()
    except ValueError:
        raise WrongGeneratorError(f'Wrong generator type: {metadata.gen_type}')

def select_keras():
    from eit_tf_workspace.keras.gen import GeneratorKeras
    return GeneratorKeras()

def select_pytorch():
    from eit_tf_workspace.pytorch.gen import GeneratorPyTorch
    return GeneratorPyTorch()


GEN_LIST={
    ListGenerators.Keras : select_keras,
    ListGenerators.Pytorch: select_pytorch
}





if __name__ == "__main__":
    from glob_utils.log.log  import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    print(ListGenerators.Keras.value)
    print(ListGenerators())




