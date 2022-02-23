


import logging
from logging import getLogger

from eit_ai.draw_data import *
from eit_ai.pytorch.tensorboard_torch import run_tensorboard
from eit_ai.pytorch.workspace import PyTorchWorkspace
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import load_samples
from eit_ai.train_utils.lists import ListPyTorchLosses, ListPyTorchOptimizers, ListPytorchDatasetHandlers, ListPytorchModelHandlers, ListPytorchModels
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.workspace import AiWorkspace


logger = getLogger(__name__)

def std_pytorch_train_pipeline(path:str= ''):
    logger.info('### Start standard pytorch training ###')

    metadata=MetaData()
    ws = PyTorchWorkspace()# Create a model generator
    ws.select_model_dataset(
        model_handler=ListPytorchModelHandlers.PytorchModelHandler,
        dataset_handler=ListPytorchDatasetHandlers.StdPytorchDatasetHandler,
        model=ListPytorchModels.StdPytorchModel,
        metadata=metadata)

    metadata.set_ouput_dir(training_name='MLP_PyTorch_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih-Xh','Yih-Yh'])
    metadata._nb_samples = 50000
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=100)
    ws.build_dataset(raw_samples, metadata)

    samples_x, samples_y = ws.extract_samples(dataset_part='train', idx_samples=None)
    plot_EIT_samples(ws.getattr_dataset('fwd_model'), samples_y, samples_x)
        
    metadata.set_4_model(epoch=100,
                         metrics=['mse'], 
                         optimizer=ListPyTorchOptimizers.Adam,
                         loss=ListPyTorchLosses.CrossEntropyLoss,
                        #  callbacks=[run_tensorboard]
                         )
    build_train_save_model(ws, metadata)

    # ws.build_model(metadata) 
    # metadata.save()# saving in case of bugs during training

    # ws.run_training(metadata)
    # ws.save_model(metadata) 
    # metadata.save() # final saving
def Conv1d_pytorch_train_pipeline(path:str= ''):
    logger.info('### Start standard pytorch training ###')

    metadata=MetaData()
    ws = PyTorchWorkspace()# Create a model generator
    ws.select_model_dataset(
        model_handler=ListPytorchModelHandlers.PytorchModelHandler,
        dataset_handler=ListPytorchDatasetHandlers.PytorchConv1dDatasetHandler,
        model=ListPytorchModels.Conv1dNet,
        metadata=metadata)

    metadata.set_ouput_dir(training_name='Conv1d_PyTorch_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih-Xh','Yih-Yh'])
    metadata._nb_samples = 50000
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=500)
    ws.build_dataset(raw_samples, metadata)

    samples_x, samples_y = ws.extract_samples(dataset_part='train', idx_samples=None)
    plot_EIT_samples(ws.getattr_dataset('fwd_model'), samples_y, samples_x)
        
    metadata.set_4_model(epoch=100,
                         metrics=['mse'], 
                         optimizer=ListPyTorchOptimizers.Adam,
                         callbacks=[run_tensorboard]
                         )
    build_train_save_model(ws, metadata)
    
def build_train_save_model(ws:AiWorkspace, metadata:MetaData)-> tuple[AiWorkspace,MetaData]:
    ws.build_model(metadata) 
    metadata.save()# saving in case of bugs during training

    ws.run_training(metadata)
    ws.save_model(metadata) 
    metadata.save()
    metadata.callbacks=True # final saving
    return ws, metadata


if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)

    debug=True

    
    if debug:
        path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    else:
        path= ''

    std_pytorch_train_pipeline(path=path)
    # Conv1d_pytorch_train_pipeline(path=path)
    plt.show()
