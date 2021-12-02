


import logging
import matplotlib.pyplot as plt
from eit_ai.keras.const import ListKerasOptimizers
from eit_ai.keras.dataset import ListKerasDatasets
from eit_ai.keras.models import ListKerasModels
from eit_ai.draw_data import plot_EIT_samples  
from eit_ai.keras.gen import GeneratorKeras
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import load_samples
from eit_ai.train_utils.gen import Generators
from eit_ai.train_utils.metadata import MetaData
from eit_ai.keras.tensorboard_k import mk_callback_tensorboard

# from eit_ai.train_utils.dataloader import 
from logging import getLogger

logger = getLogger(__name__)


def std_keras_train_pipeline(path:str= ''):
    logger.info('### Start standard keras training ###')

    metadata=MetaData()
    gen = GeneratorKeras()# Create a model generator
    gen.select_model_dataset(
        model_type=ListKerasModels.StdKerasModel,
        dataset_type=ListKerasDatasets.StdDataset,
        metadata=metadata)

    metadata.set_ouput_dir(training_name='Std_keras_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih-Xh','Yih-Yh'])
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=1000)
    gen.build_dataset(raw_samples, metadata)

    samples_x, samples_y = gen.extract_samples(dataset_part='train', idx_samples=None)
    plot_EIT_samples(gen.getattr_dataset('fwd_model'), samples_y, samples_x)
        
    metadata.set_4_model(
        epoch=100,
        callbacks=[mk_callback_tensorboard(metadata)],
        metrics=['mse'],
        optimizer=ListKerasOptimizers.Adam)

    build_train_save_model(gen, metadata)

def std_auto_pipeline(path=''):
    logger.info('### Start standard autokeras training ###')

    metadata=MetaData()
    gen = GeneratorKeras()# Create a model generator
    gen.select_model_dataset(
        model_type=ListKerasModels.StdAutokerasModel,
        dataset_type=ListKerasDatasets.StdDataset,
        metadata=metadata)
    metadata.set_ouput_dir(training_name='Std_autokeras_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih-Xh','Yih-Yh'])
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=1000,)
    gen.build_dataset(raw_samples, metadata)

    samples_x, samples_y = gen.extract_samples(dataset_part='train', idx_samples=None)
    plot_EIT_samples(gen.getattr_dataset('fwd_model'), samples_y, samples_x)
        
    metadata.set_4_model(
        epoch=2,
        callbacks=[mk_callback_tensorboard(metadata)],
        metrics=['mse'],
        max_trials_autokeras=2)

    build_train_save_model(gen, metadata)

def build_train_save_model(gen:Generators, metadata:MetaData)-> tuple[Generators,MetaData]:
    gen.build_model(metadata) 
    metadata.save()# saving in case of bugs during training

    gen.run_training(metadata)
    gen.save_model(metadata) 
    metadata.save() # final saving
    return gen, metadata

# def normalize_image(image, label):
#     return np.resize(image, (-1,image.shape[0])), label

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)

    debug=True

    if debug:
        path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    else:
        path= ''

    # std_keras_train_pipeline(path=path)

    std_auto_pipeline(path=path)
    plt.show()
