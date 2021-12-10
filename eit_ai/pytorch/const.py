

from enum import Enum
from torch import nn, optim
from eit_ai.train_utils.lists import ListPyTorchLosses, ListPyTorchOptimizers

PYTORCH_MODEL_SAVE_FOLDERNAME= 'pytorch_model.pth'
################################################################################
# Optimizers
################################################################################


PYTORCH_OPTIMIZER={
    ListPyTorchOptimizers.Adam: optim.Adam
}
################################################################################
# Losses
################################################################################


PYTORCH_LOSS={
    ListPyTorchLosses.MSELoss: nn.MSELoss
}

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
