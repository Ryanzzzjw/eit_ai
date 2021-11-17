


from eit_tf_workspace.train_utils.gen import Generators
from eit_tf_workspace.train_utils.lists import ListGenerators
from eit_tf_workspace.keras.gen import GeneratorKeras
from eit_tf_workspace.pytorch.gen import GeneratorPyTorch
from eit_tf_workspace.train_utils.metadata import MetaData


GEN_LIST={
    ListGenerators.Keras : GeneratorKeras,
    ListGenerators.Pytorch: GeneratorPyTorch
}

class WrongGeneratorError(Exception):
    """"""

def select_gen(metadata:MetaData) -> Generators:

    try:
        return GEN_LIST[ListGenerators(metadata.gen_type)]()
    except ValueError:
        raise WrongGeneratorError(f'Wrong generator type: {metadata.gen_type}')

if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    print(ListGenerators.Keras.value)
    print(ListGenerators())




