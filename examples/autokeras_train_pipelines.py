


import logging
import matplotlib.pyplot as plt
import numpy as np
from eit_ai.keras.const import ListKerasOptimizers
from eit_ai.keras.dataset import ListKerasDatasetHandlers
from eit_ai.keras.models import ListKerasModelHandlers
from eit_ai.draw_data import plot_EIT_samples  
from eit_ai.keras.workspace import KerasWorkspace
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import load_samples
from eit_ai.train_utils.lists import ListKerasModels
from eit_ai.train_utils.workspace import AiWorkspace
from eit_ai.train_utils.metadata import MetaData
from eit_ai.keras.tensorboard_k import mk_callback_tensorboard
from keras_train_pipelines import build_train_save_model

# from eit_ai.train_utils.dataloader import 
import logging

logger = logging.getLogger(__name__)

def std_auto_pipeline(path=''):
    logger.info('### Start standard autokeras training ###')

    metadata=MetaData()
    gen = KerasWorkspace()# Create a model generator
    gen.select_model_dataset(
        model_handler=ListKerasModelHandlers.AutokerasModelHandler,
        dataset_handler=ListKerasDatasetHandlers.KerasDatasetHandler,
        model=ListKerasModels.StdAutokerasModel,
        metadata=metadata)
    metadata.set_ouput_dir(training_name='Std_autokeras_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih-Xh/Xh','Yih-Yh'])
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=128)
    gen.build_dataset(raw_samples, metadata)

    samples_x, samples_y = gen.extract_samples(dataset_part='train', idx_samples=None)
    plot_EIT_samples(gen.getattr_dataset('fwd_model'), samples_y, samples_x)
        
    metadata.set_4_model(
        epoch=100,
        callbacks=[mk_callback_tensorboard(metadata)],
        metrics=['mse'],
        max_trials_autokeras=100)

    build_train_save_model(gen, metadata)


if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)

    std_auto_pipeline()
    plt.show()
