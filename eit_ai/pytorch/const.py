

from enum import Enum
from torch import nn, optim
from eit_ai.train_utils.lists import ListLosses, ListOptimizers



PYTORCH_MODEL_SAVE_FOLDERNAME='torch_model'

################################################################################
# Optimizers
################################################################################

class PytorchOptimizers(Enum):
    Adam='Adam'

PYTORCH_OPTIMIZER={
    PytorchOptimizers.Adam: optim.Adam
}
################################################################################
# Losses
################################################################################

class PytorchLosses(Enum):
    MSELoss='MSELoss'

PYTORCH_LOSS={
    PytorchLosses.MSELoss: nn.MSELoss
}

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
