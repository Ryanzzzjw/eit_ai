
import tensorflow.keras as keras
from eit_ai.train_utils.lists import ListKerasLosses, ListKerasOptimizers

KERAS_MODEL_SAVE_FOLDERNAME='keras_model'

################################################################################
# Keras Optimizers
################################################################################
""" Dictionary listing all Keras optimizers available
"""
KERAS_OPTIMIZERS={
    ListKerasOptimizers.Adam:keras.optimizers.Adam
}
################################################################################
# Keras Losses
################################################################################
""" Dictionary listing all Keras losses available
"""
KERAS_LOSSES={
    ListKerasLosses.CategoricalCrossentropy:keras.losses.CategoricalCrossentropy,
    ListKerasLosses.MeanSquaredError:keras.losses.MeanSquaredError
}

if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
