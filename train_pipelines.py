


from tensorboard import data
import tensorflow.keras as keras
from tensorflow.python.keras.losses import MSE


# from modules.load_mat_files import *
from modules.dataset import *
from modules.draw_data import *
from modules.train_models import *
from modules.train_utils import *
import modules.constants as const

<<<<<<< HEAD

=======
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
def std_training_pipeline(verbose=False, path= ''):
    
    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
<<<<<<< HEAD
    train_inputs.init_ouput(training_name='Std_keras', append_time= True) # set the naem of the current training and cretae directory for ouputs

=======
    train_inputs.init_ouput(training_name='Autokeras_2Dcoarse_adad_c3_mm_', append_time= True) # set the naem of the current training and cretae directory for ouputs
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
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
<<<<<<< HEAD
                                    epoch=3,
=======
                                    epoch=500,
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
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

<<<<<<< HEAD
def std_auto_pipeline(verbose=False, path=''):

    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
    train_inputs.init_ouput(training_name='Autokeras', append_time= True) # set the naem of the current training and cretae directory for ouputs

    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose)
=======
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
     "layers": [2,3,4],
     "sizes": [128,256,512],
     "dropouts": [0,0.2]
    } 

    dense_layers = param["layers"]
    layer_sizes = param["sizes"]
    dropouts = param["dropouts"]


    for dropout in dropouts:
        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                
                train_inputs.init_ouput(training_name=f'OPT_{dropout}dropout_{dense_layer}layer_{layer_size}layers_size', append_time= True)
                NAME = "{}-dropout-{}-layers-{}layer_size".format(dropout,dense_layer,layer_size, int(time.time()))
                tensorboard = mk_callback_tensorboard(NAME)
                train_inputs.set_idx_samples(save_idx_samples_2matfile(raw_data,dataset,train_inputs.time))  # extract and save th idx of the samples
                

                train_inputs.set_values4model(  model_func=gen.opt_keras,
                                                dataset=dataset,
                                                epoch=500,
                                                callbacks=[tensorboard],
                                                metrics=[MSE])
                train_inputs.layer_size=layer_size
                train_inputs.layer_nb=dense_layer
                train_inputs.dropout=dropout
                
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
    train_inputs.init_ouput(training_name='Autokeras_2Dcoarse_2nd_adad_c3_mm', append_time= True) # set the naem of the current training and cretae directory for ouputs
    
    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih-Xh','Yih-Yh'],verbose=verbose)
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_values4dataloader(batch_size=1000,use_tf_dataset=False)
    dataset = dataloader(raw_data,verbose=verbose, train_inputs=train_inputs)

    train_inputs.set_idx_samples(save_idx_samples_2matfile(raw_data,dataset,train_inputs.time))
        
<<<<<<< HEAD
    if verbose:
        samples_x, samples_y = extract_samples(dataset, dataset_part='train', idx_samples=None)
        plot_EIT_samples(dataset.fwd_model,samples_y,samples_x)
=======
    #if verbose:
    #   samples_x, samples_y = extract_samples(dataset, dataset_part='train', idx_samples=None)
    #  plot_EIT_samples(dataset.fwd_model,samples_y,samples_x)
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
    
    tensorboard = mk_callback_tensorboard(train_inputs)

    train_inputs.set_values4model(  model_func=gen.std_autokeras,
                                    dataset=dataset,
<<<<<<< HEAD
                                    epoch=1,
                                    callbacks=[tensorboard],
                                    max_trials_autokeras=1)
=======
                                    epoch=500,
                                    callbacks=[tensorboard],
                                    max_trials_autokeras=100)
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
    
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
<<<<<<< HEAD
if __name__ == "__main__":
    debug=True
=======

if __name__ == "__main__":
    debug=False
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
    
    if debug:
        path='datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    else:
        path= ''

    #std_training_pipeline(verbose=True, path=path)
<<<<<<< HEAD

    std_auto_pipeline(verbose=True, path=path)
=======
    std_opt_pipeline(verbose=True, path=path)
    # std_auto_pipeline(verbose=True, path=path)
>>>>>>> c03e94d295b7ed88614f33427c4ac5e9cf7b069e
    plt.show()
