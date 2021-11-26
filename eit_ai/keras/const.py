
import tensorflow.keras as keras
from eit_ai.train_utils.lists import ListLosses, ListOptimizers



KERAS_MODEL_SAVE_FOLDERNAME='keras_model'

################################################################################
# Optimizers
################################################################################

class KerasOptimizers(ListOptimizers):
    Adam='Adam'

KERAS_OPTIMIZER={
    KerasOptimizers.Adam:keras.optimizers.Adam
}
################################################################################
# Losses
################################################################################

class KerasLosses(ListLosses):
    CategoricalCrossentropy='CategoricalCrossentropy'

KERAS_LOSS={
    KerasLosses.CategoricalCrossentropy:keras.losses.CategoricalCrossentropy
}

if __name__ == "__main__":
    from eit_ai.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)