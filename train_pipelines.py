


from tensorboard import data
import tensorflow.keras as keras
from tensorflow.python.keras.losses import MSE


# from modules.load_mat_files import *
from modules.dataset import *
from modules.draw_data import *
from modules.train_models import *
from modules.train_utils import *
import modules.constants as const

def std_training_pipeline(verbose=False, path= ''):
    
    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
    train_inputs.init_ouput(training_name='Autokeras_2Dcoarse_adad_c3_mm_', append_time= True) # set the naem of the current training and cretae directory for ouputs
    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose, time=train_inputs.time)
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_values4dataloader(batch_size=1000, use_tf_dataset=False)
    dataset = dataloader(raw_data,verbose=verbose, train_inputs=train_inputs)

    train_inputs.set_idx_samples(save_idx_samples_2matfile(raw_data,dataset,train_inputs.time))  # extract and save th idx of the samples

    if verbose:
        samples_x, samples_y = extract_samples(dataset, dataset_part='train', idx_samples=None)
        plot_EIT_samples(dataset.fwd_model,samples_y,samples_x)
        
    tensorboard = mk_callback_tensorboard(train_inputs)

    train_inputs.set_values4model(  model_func=gen.std_keras,
                                    dataset=dataset,
                                    epoch=500,
                                    callbacks=[tensorboard],
                                    metrics=[MSE])
    
    gen.select_model(train_inputs)
    gen.compile_model(train_inputs=train_inputs)

    train_inputs.save()# first saving in case of bugs

    # Train the model
    gen.mk_fit(dataset,train_inputs=train_inputs)
    
    # Save the trained model
    train_inputs.model_saving_path=gen.save_model(path=train_inputs.ouput_dir) 

    train_inputs.save()# final saving in case of bugs

def std_opt_pipeline(verbose=False, path= ''):
    
    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
     # set the naem of the current training and cretae directory for ouputs
    # get data from matlab
    train_inputs.init_ouput(training_name=f'INIT_', append_time= True)
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose, time=get_date_time())
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_values4dataloader(batch_size=1000, use_tf_dataset=False)
    dataset = dataloader(raw_data,verbose=verbose, train_inputs=train_inputs)
    if verbose:
        samples_x, samples_y = extract_samples(dataset, dataset_part='train', idx_samples=None)
        plot_EIT_samples(dataset.fwd_model,samples_y,samples_x)

    # tensorboard = mk_callback_tensorboard(train_inputs)
    param = {
     "layers": [3],
     "sizes": [256,512,1024,2048],
     "dropouts": [0,1]
    } 

    dense_layers = param["layers"]
    layer_sizes = param["sizes"]
    dropouts = param["dropouts"]

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            
            train_inputs.init_ouput(training_name=f'OPT_var_in_middle_layers{dense_layer}layer_{layer_size}1st_hidden_layers_size', append_time= True)
            NAME = "{}-layers-{}layer_size".format(dense_layer,layer_size, int(time.time()))
            tensorboard = mk_callback_tensorboard(NAME)
            train_inputs.set_idx_samples(save_idx_samples_2matfile(raw_data,dataset,train_inputs.time))  # extract and save th idx of the samples
            

            train_inputs.set_values4model(  model_func=gen.opt_keras,
                                            dataset=dataset,
                                            epoch=500,
                                            callbacks=[tensorboard],
                                            metrics=[MSE])
            train_inputs.layer_size=layer_size
            train_inputs.layer_nb=dense_layer
            # train_inputs.dropout=dropout
            
            gen.select_model(train_inputs)
            gen.compile_model(train_inputs=train_inputs)

            train_inputs.save()# first saving in case of bugs

        # Train the model
            gen.mk_fit(dataset,train_inputs=train_inputs)
        # Save the trained model
            train_inputs.model_saving_path=gen.save_model(path=train_inputs.ouput_dir) 

            train_inputs.save()# final saving in case of bugs

def std_auto_pipeline(verbose=False, path=''):

    gen = ModelGenerator()# Create a model generator
    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
    train_inputs.init_ouput(training_name='Autokeras_2D_opop_2nd_c3_mm', append_time= True) # set the naem of the current training and cretae directory for ouputs
    
    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih-Xh','Yih-Yh'],verbose=verbose)
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_values4dataloader(batch_size=1000,use_tf_dataset=False)
    dataset = dataloader(raw_data,verbose=verbose, train_inputs=train_inputs)

    train_inputs.set_idx_samples(save_idx_samples_2matfile(raw_data,dataset,train_inputs.time))
        
    #if verbose:
    #   samples_x, samples_y = extract_samples(dataset, dataset_part='train', idx_samples=None)
    #  plot_EIT_samples(dataset.fwd_model,samples_y,samples_x)
    
    tensorboard = mk_callback_tensorboard(train_inputs)

    train_inputs.set_values4model(  model_func=gen.std_autokeras,
                                    dataset=dataset,
                                    epoch=500,
                                    callbacks=[tensorboard],
                                    max_trials_autokeras=100)
    
    gen.select_model(train_inputs)
    gen.compile_model(train_inputs=train_inputs)

    train_inputs.save()

    # Train the model
    gen.mk_fit(dataset,train_inputs=train_inputs)
    # ######## THIS IS WORKING ############
    # perm_nn = gen.mk_prediction(dataset.test) 
    # print('perm_NN', perm_nn.shape)
    # #####################################
    # Save the trained model
    train_inputs.model_saving_path=gen.save_model(path=train_inputs.ouput_dir, save_summary=False)
    train_inputs.save()

    # gen1 = ModelGenerator()
    # gen1.load_model(train_inputs.model_saving_path)
    # print(gen1.model.summary())    
    # perm_nn = gen1.mk_prediction(dataset.test) 
    # print('perm_NN', perm_nn.shape)
    
def normalize_image(image, label):
    return np.resize(image, (-1,image.shape[0])), label

if __name__ == "__main__":
    debug=False
    
    if debug:
        path='/data/Metz/Python/eit_tf_workspace/datasets/20211026_130736_2D_16e_adad_cell1_SNR20dB_50k_Mantas_dataset/2D_16e_adad_cell1_SNR20dB_50k_Mantas_infos2py.mat'
    else:
        path= ''

    # std_training_pipeline(verbose=True, path=path)
    std_opt_pipeline(verbose=True, path=path)
    # std_auto_pipeline(verbose=True, path=path)
    plt.show()
