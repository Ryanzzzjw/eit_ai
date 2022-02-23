

from enum import Enum
from torch import nn, optim
from eit_ai.train_utils.lists import ListPyTorchLosses, ListPyTorchOptimizers

PYTORCH_MODEL_SAVE_FOLDERNAME= 'pytorch_model.pth'
################################################################################
# PyTorch Optimizers
################################################################################
""" Dictionary listing all PyTorch optimizers available
"""

PYTORCH_OPTIMIZER={
    ListPyTorchOptimizers.Adam: optim.Adam,
    ListPyTorchOptimizers.SGD: optim.SGD
}
################################################################################
# PyTorch Losses
################################################################################
""" Dictionary listing all PyTorch losses available
"""

PYTORCH_LOSS={
    ListPyTorchLosses.MSELoss: nn.MSELoss,
    ListPyTorchLosses.CrossEntropyLoss: nn.CrossEntropyLoss,
}

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
