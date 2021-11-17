


import logging
from eit_tf_workspace.keras.dataset import KerasDatasets
from eit_tf_workspace.keras.models import KerasModels
from eit_tf_workspace.draw_data import *
from eit_tf_workspace.keras.gen import GeneratorKeras
from eit_tf_workspace.raw_data.matlab import MatlabSamples
from eit_tf_workspace.raw_data.raw_samples import load_samples
from eit_tf_workspace.train_utils.metadata import MetaData
from eit_tf_workspace.keras.tensorboard_k import mk_callback_tensorboard

# from eit_tf_workspace.train_utils.dataloader import 
from logging import getLogger

logger = getLogger(__name__)

def std_keras_train_pipeline(path:str= ''):
    logger.info('### Start standard keras training ###')

    metadata=MetaData()
    gen = GeneratorKeras()# Create a model generator
    gen.select_model_dataset(
        model_type=KerasModels.StdKerasModel,
        dataset_type=KerasDatasets.StdDataset,
        metadata=metadata)

    metadata.set_ouput_dir(training_name='Std_keras_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih-Xh','Yih-Yh'])
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=1000)
    gen.build_dataset(raw_samples, metadata)

    samples_x, samples_y = gen.extract_samples(dataset_part='train', idx_samples=None)
    plot_EIT_samples(gen.getattr_dataset('fwd_model'), samples_y, samples_x)
        
    metadata.set_4_model(
        epoch=2,
        callbacks=[mk_callback_tensorboard(metadata)],
        metrics=['mse'])

    gen.build_model(metadata) 
    metadata.save()# saving in case of bugs during training

    gen.run_training(metadata)
    gen.save_model(metadata) 
    metadata.save() # final saving

def std_auto_pipeline(verbose=False, path=''):
    logger.info('### Start standard autokeras training ###')

    gen = GeneratorKeras()# Create a model generator

    train_inputs=MetaData() # create class to managed training variables (saving them,....)
    train_inputs.set_ouput_dir(training_name='Autokeras', append_date_time= True) # set the naem of the current training and cretae directory for ouputs

    # get data from matlab 
    raw_data=get_matlab_dataset(file_path=path, data_sel= ['Xih','Yih'],verbose=verbose)
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_4_dataset(batch_size=1000,use_tf_dataset=False)
    dataset = create_dataset(raw_data,verbose=verbose, metadata=train_inputs)

    train_inputs.set_idx_samples_file(save_idx_samples_2matfile(raw_data,dataset,train_inputs.time))
        
    if verbose:
        samples_x, samples_y = extract_samples(dataset, dataset_part='train', idx_samples=None)
        plot_EIT_samples(dataset.fwd_model,samples_y,samples_x)
    
    tensorboard = mk_callback_tensorboard(train_inputs)

    train_inputs.set_4_model(  model_type=gen.std_autokeras,
                                    dataset=dataset,
                                    epoch=1,
                                    callbacks=[tensorboard],
                                    max_trials_autokeras=1)
    
    gen.build_model(train_inputs)
    gen.prepare_model(train_inputs=train_inputs)

    train_inputs.save()

    # Train the model
    gen.run_training(dataset,train_inputs=train_inputs)
    train_inputs.model_saving_path=gen.save_model(path=train_inputs.ouput_dir, save_summary=False)
    train_inputs.save()

    
# def normalize_image(image, label):
#     return np.resize(image, (-1,image.shape[0])), label

if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)

    debug=True

    
    if debug:
        path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    else:
        path= ''

    std_keras_train_pipeline(path=path)

    #std_auto_pipeline(verbose=True, path=path)
    plt.show()
