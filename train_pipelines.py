


import tensorflow.keras as keras


# from modules.load_mat_files import *
from modules.dataset import *
from modules.draw_data import *
from modules.train_models import *
from modules.train_utils import *


def std_training_pipeline(verbose=False, path= ''):
    
    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
    train_inputs.init_ouput(training_name='Test_keras', append_time= True) # set the naem of the current training and cretae directory for ouputs

    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose)
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_values4dataloader(batch_size=1000)
    dataset = dataloader(raw_data,verbose=verbose, train_inputs=train_inputs)

    train_inputs.idx_samples_file= save_idx_samples_2matfile(raw_data,dataset)  # extract and save th idx of the samples

    if verbose:
        if dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in dataset.train.as_numpy_iterator():
                print(inputs[0])
                print('label of this input is', outputs[0])
                plot_EIT_samples(dataset.fwd_model, outputs[0],inputs[0])
                
                break

    tensorboard = mk_callback_tensorboard(train_inputs)

    train_inputs.set_values4model(  model_func=gen.std_keras,
                                    dataset=dataset,
                                    epoch=1,
                                    callbacks=[tensorboard])
    
    gen.select_model(train_inputs)
    gen.compile_model(train_inputs=train_inputs)

    train_inputs.save()

    # Train the model
    gen.mk_fit(dataset,train_inputs=train_inputs)
    
    # Save the trained model
    gen.save_model(path=train_inputs.ouput_dir)             

def std_auto_pipeline(verbose=False, path=''):

    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
    train_inputs.init_ouput(training_name='Autokeras', append_time= True) # set the naem of the current training and cretae directory for ouputs

    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose)
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_values4dataloader(batch_size=32,use_tf_dataset=False)
    dataset = dataloader(raw_data,verbose=verbose, train_inputs=train_inputs)

    # train_inputs.idx_samples_file= save_idx_samples_2matfile(raw_data,dataset)
        
    if not verbose:
        if dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in dataset.train.as_numpy_iterator():
                print(inputs[0])
                print('label of this input is', outputs[0])
               #plot_EIT_samples(dataset.fwd_model, outputs[0],inputs[0])
                break
    
    tensorboard = mk_callback_tensorboard(train_inputs)

    train_inputs.set_values4model(  model_func=gen.std_autokeras,
                                    dataset=dataset,
                                    epoch=2,
                                    callbacks=[tensorboard],
                                    max_trials_autokeras=2)
    
    gen.select_model(train_inputs)
    gen.compile_model(train_inputs=train_inputs)

    train_inputs.save()

    # Train the model
    gen.mk_fit(dataset,train_inputs=train_inputs)
    
    # Save the trained model
    gen.save_model(path=train_inputs.ouput_dir) 

       

if __name__ == "__main__":
    debug=True
    
    if debug:
        path='datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    else:
        path= ''

    #std_training_pipeline(verbose=True, path=path)

    std_auto_pipeline(verbose=True, path=path)
    plt.show()
