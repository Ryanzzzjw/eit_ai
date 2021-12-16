


import logging
# from eit_ai.train_utils.dataloader import 
from logging import getLogger

from eit_ai.draw_data import *
from eit_ai.pytorch.workspace import PyTorchWorkspace
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import load_samples
from eit_ai.train_utils.lists import ListPytorchDatasetHandlers, ListPytorchModelHandlers, ListPytorchModels
from eit_ai.train_utils.metadata import MetaData


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

    metadata.set_ouput_dir(training_name='Std_PyTorch_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih-Xh','Yih-Yh'])
    metadata._nb_samples = 20000
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=1000)
    ws.build_dataset(raw_samples, metadata)

    samples_x, samples_y = ws.extract_samples(dataset_part='train', idx_samples=None)
    plot_EIT_samples(ws.getattr_dataset('fwd_model'), samples_y, samples_x)
        
    metadata.set_4_model(epoch=10,metrics=['mse'])

    ws.build_model(metadata) 
    metadata.save()# saving in case of bugs during training

    ws.run_training(metadata)
    ws.save_model(metadata) 
    metadata.save() # final saving


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

    # std_keras_train_pipeline(path=path)

    std_pytorch_train_pipeline()
    plt.show()
