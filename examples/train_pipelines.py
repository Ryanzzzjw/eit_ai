


from tensorboard import data
import tensorflow.keras as keras
from tensorflow.python.keras.losses import MSE


# from modules.load_mat_files import *
from eit_tf_workspace.dataset import *
from eit_tf_workspace.draw_data import *
from eit_tf_workspace.train_models import *
from eit_tf_workspace.train_utils import *
import eit_tf_workspace.constants as const


def std_training_pipeline(verbose=False, path= ''):
    
    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
    train_inputs.init_ouput(training_name='Std_keras_epoch500', append_time= True) # set the naem of the current training and cretae directory for ouputs

    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih-Xh','Yih-Yh'],verbose=verbose, time=train_inputs.time)
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

def std_auto_pipeline(verbose=False, path=''):

    gen = ModelGenerator()# Create a model generator

    train_inputs=TrainInputs() # create class to managed training variables (saving them,....)
    train_inputs.init_ouput(training_name='Autokeras', append_time= True) # set the naem of the current training and cretae directory for ouputs

    # get data from matlab 
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose)
    train_inputs.data_select= raw_data.data_sel

    # data preprocessing
    train_inputs.set_values4dataloader(batch_size=1000,use_tf_dataset=False)
    dataset = dataloader(raw_data,verbose=verbose, train_inputs=train_inputs)

    train_inputs.set_idx_samples(save_idx_samples_2matfile(raw_data,dataset,train_inputs.time))
        
    if verbose:
        samples_x, samples_y = extract_samples(dataset, dataset_part='train', idx_samples=None)
        plot_EIT_samples(dataset.fwd_model,samples_y,samples_x)
    
    tensorboard = mk_callback_tensorboard(train_inputs)

    train_inputs.set_values4model(  model_func=gen.std_autokeras,
                                    dataset=dataset,
                                    epoch=1,
                                    callbacks=[tensorboard],
                                    max_trials_autokeras=1)
    
    gen.select_model(train_inputs)
    gen.compile_model(train_inputs=train_inputs)

    train_inputs.save()

    # Train the model
    gen.mk_fit(dataset,train_inputs=train_inputs)
    train_inputs.model_saving_path=gen.save_model(path=train_inputs.ouput_dir, save_summary=False)
    train_inputs.save()

    
def normalize_image(image, label):
    return np.resize(image, (-1,image.shape[0])), label
if __name__ == "__main__":
    debug=True
    
    if debug:
        path=''
    else:
        path= ''

    std_training_pipeline(verbose=True, path=path)

    #std_auto_pipeline(verbose=True, path=path)
    plt.show()
