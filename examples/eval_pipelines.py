

from enum import auto
from sys import modules

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import reshape
from scipy.io import loadmat
from tensorflow.python.keras.metrics import FalseNegatives

import eit_tf_workspace.constants as const
from eit_tf_workspace.dataset import *
from eit_tf_workspace.draw_data import *
from eit_tf_workspace.eval_utils import error_eval
from eit_tf_workspace.path_utils import get_dir, mk_ouput_dir, verify_file
from eit_tf_workspace.train_models import *
from eit_tf_workspace.utils.log import main_log
from eit_tf_workspace.mat_files import MatlabDataSet, get_MalabDataSet, save_idx_samples_2matfile
import random

from logging import getLogger
logger = getLogger(__name__)


def f(x):
    return np.int(x)
f2 = np.vectorize(f)

def std_eval_pipeline(verbose=False):

    title= 'Select directory of model to evaluate'
    path_dir=get_dir(title=title)

    #path_dir='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Std_keras_20211007_144918'
    if not path_dir:
        return
    # read train inputs instead

    training_settings=TrainMetaData()
    training_settings.read(os.path.join(path_dir,const.TRAIN_INPUT_FILENAME))
    
    path_pkl=training_settings.dataset_src_file[1]
    data_sel= training_settings.data_select
    # Data loading
    raw_data=get_MalabDataSet(file_path=path_pkl, data_sel= data_sel,verbose=verbose)#, type2load='.pkl')
    eval_dataset = dataloader(raw_data, use_tf_dataset=False,verbose=verbose, train_inputs=training_settings)
    
    if verbose:
        print(eval_dataset.use_tf_dataset)
        if eval_dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in eval_dataset.test.as_numpy_iterator():
                print('samples size:',inputs.shape, outputs.shape)
                # Print the first element and the label
                # print(inputs[0])
                # print('label of this input is', outputs[0])
                if eval_dataset.batch_size:
                    plot_EIT_samples(eval_dataset.fwd_model, outputs[0], inputs[0])
                else:
                    plot_EIT_samples(eval_dataset.fwd_model, outputs, inputs)
                break
    

    _, perm_real=extract_samples(eval_dataset, dataset_part='test', idx_samples='all', elem_idx = 1)

    print('\nperm_real',perm_real.shape)
       # Load model
    gen = ModelGenerator()
    gen.load_model(training_settings.model_saving_path)
    print(gen.model.summary())

    gen.save_model('test', True)

    # make predictions
    steps = eval_dataset.test_len//eval_dataset.batch_size if eval_dataset.batch_size else None
    perm_nn = gen.mk_prediction(eval_dataset.test,steps=steps)
    # model_evaluation= gen.model.evaluate(eval_dataset.test,steps=steps)

    # print('model_evaluation',model_evaluation)
    
    print('perm_NN', perm_nn.shape)
    # plot_EIT_samples(eval_dataset.fwd_model, perm_NN[0].T, inputs[0])

    #  eval predictions vs
    # load Eidors samples
    # perm_eidors_path= training_settings.idx_samples_file[0].replace(const.EXT_IDX_FILE, const.EXT_EIDORS_SOLVING_FILE)
    # tmp= MatlabDataSet(verbose=True)
    # if verify_file(perm_eidors_path, const.EXT_MAT):
    #     tmp.mk_dataset_from_matlab(path= perm_eidors_path, only_get_samples_EIDORS=True)
    #     perm_eidors= tmp.samples_EIDORS['elem_data'].T # matlab samples are columnwise sorted
    #     perm_eidors_n= tmp.samples_EIDORS['elem_data_n'].T

    # else:
    #     print('############ no data from EIDORS available ###############')
    #     perm_eidors= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])
    #     perm_eidors=tf.keras.utils.normalize(perm_eidors, axis=0).astype("float32")
    #     perm_eidors_n= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])

    
    results= list()
    results.append(error_eval(perm_real, perm_nn[:perm_real.shape[0],:], verbose=False, axis_samples=0, info='Results NN'))
    #results.append(error_eval(perm_real, perm_eidors[:perm_real.shape[0],:], verbose=False, axis_samples=0, info='Results Eidors'))
    
    #Generate 5 random numbers between 10 and 30
    randnlist = random.sample(range(eval_dataset.test_len), 5)
    plot_real_NN_EIDORS(eval_dataset.fwd_model, perm_real[randnlist,:].T, perm_nn[randnlist,:].T)
    
    plot_eval_results(results, axis='linear')

def std_eidors_eval_pipeline(verbose=False):

    title= 'Select directory of model to evaluate'
    path_dir=get_dir(title=title)

    #path_dir='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Std_keras_20211007_144918'
    if not path_dir:
        return
    # read train inputs instead

    training_settings=TrainMetaData()
    training_settings.read(os.path.join(path_dir,const.TRAIN_INPUT_FILENAME))

    # with open(os.path.join(path_dir,'dataset_src_file.txt')) as f:
    #     path_pkl=f.readline().replace('\n','')
    #     path_pkl=f.readline().replace('\n','')
    
    path_pkl=training_settings.dataset_src_file[1]
    data_sel= training_settings.data_select
    # Data loading
    raw_data=get_MalabDataSet(file_path=path_pkl, data_sel= data_sel,verbose=verbose)#, type2load='.pkl')

    idx_samples_path= training_settings.idx_samples_file[0]
    tmp= MatlabDataSet(verbose=True)
    tmp.load_dataset_from_mat_file(file_path= idx_samples_path, only_get_samples_EIDORS=True)
    print(tmp.samples_EIDORS['idx_train'].flatten() ,tmp.samples_EIDORS['idx_train'].flatten().shape)
    idx=[]
    idx.append(f2(tmp.samples_EIDORS['idx_train'].flatten()))
    idx.append(f2(tmp.samples_EIDORS['idx_val'].flatten() ))
    idx.append(f2(tmp.samples_EIDORS['idx_test'].flatten() ))
    print(idx)



    eval_dataset = dataloader(raw_data, use_tf_dataset=False,verbose=verbose, train_inputs=training_settings, idx=idx)
    #save_idx_samples_2matfile(raw_data,eval_dataset,get_date_time())
    if verbose:
        print(eval_dataset.use_tf_dataset)
        if eval_dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in eval_dataset.test.as_numpy_iterator():
                print('samples size:',inputs.shape, outputs.shape)
                # Print the first element and the label
                # print(inputs[0])
                # print('label of this input is', outputs[0])
                if eval_dataset.batch_size:
                    plot_EIT_samples(eval_dataset.fwd_model, outputs[0], inputs[0])
                else:
                    plot_EIT_samples(eval_dataset.fwd_model, outputs, inputs)
                break
    
    

    _, perm_real=extract_samples(eval_dataset, dataset_part='test', idx_samples='all'  , elem_idx = 1)

    print('\nperm_real',perm_real.shape)
       # Load model
    gen = ModelGenerator()
    gen.load_model(training_settings.model_saving_path)
    print(gen.model.summary())

    gen.save_model('test', True)

    # make predictions
    steps = eval_dataset.test_len//eval_dataset.batch_size if eval_dataset.batch_size else None
    perm_nn = gen.mk_prediction(eval_dataset.test,steps=steps)
    # model_evaluation= gen.model.evaluate(eval_dataset.test,steps=steps)

    # print('model_evaluation',model_evaluation)
    
    print('perm_NN', perm_nn.shape)
    # plot_EIT_samples(eval_dataset.fwd_model, perm_NN[0].T, inputs[0])

    #  eval predictions vs
    # load Eidors samples
    perm_eidors_path= os.path.join(os.path.split(training_settings.idx_samples_file[0])[0], 'elems_solved.mat')
    tmp= MatlabDataSet(verbose=True)
    if verify_file(perm_eidors_path, const.EXT_MAT):
        tmp.load_dataset_from_mat_file(file_path= perm_eidors_path, only_get_samples_EIDORS=True)
        perm_eidors= tmp.samples_EIDORS['elem_data'].T # matlab samples are columnwise sorted
        perm_eidors=scale_prepocess(perm_eidors, True)
        print(perm_eidors, perm_eidors.shape)
        #perm_eidors_n= tmp.samples_EIDORS['elem_data_n'].T
    else:
        print('############ no data from EIDORS available ###############')
        perm_eidors= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])
        perm_eidors=tf.keras.utils.normalize(perm_eidors, axis=0).astype("float32")
        #perm_eidors_n= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])

    
    nb_eidors= perm_eidors.shape[0]

    results= list()
    results.append(error_eval(perm_real[:nb_eidors,:], perm_nn[:nb_eidors,:], verbose=False, axis_samples=0, info='Results NN'))
    results.append(error_eval(perm_real[:nb_eidors,:], perm_eidors[:nb_eidors,:], verbose=False, axis_samples=0, info='Results Eidors'))
    
    #Generate 5 random numbers between 10 and 30
    #randnlist = random.sample(range(nb_eidors), 3)
    randnlist=[range(5)] 
    plot_real_NN_EIDORS(eval_dataset.fwd_model, perm_real[randnlist,:].T, perm_nn[randnlist,:].T, perm_eidors[randnlist,:].T)
    
    plot_eval_results(results, axis='linear')
    

if __name__ == "__main__":
    main_log()
    # a= TrainInputs()
    # a.read('E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Std_keras_20211006_165901/train_inputs.txt')


    # path_pkl= 'datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    # print(verify_file(path_pkl, extension=".pkl", debug=True))
    # get_XY_from_MalabDataSet(path=path_pkl, data_sel= ['Xih','Yih'],verbose=True)
    # std_eval_pipeline(verbose=True)
    std_eidors_eval_pipeline(verbose=True)

    plt.show()
    
