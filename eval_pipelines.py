

from math import perm
from os import getcwd
from sys import modules

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import reshape
from scipy.io import loadmat
from tensorflow.python.keras.metrics import FalseNegatives

import modules.constants as const
from modules.dataset import *
from modules.draw_data import *
from modules.eval_utils import error_eval
from modules.load_mat_files import *
from modules.path_utils import get_dir, mk_ouput_dir, verify_file
from modules.train_models import *
import random

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

    training_settings=TrainInputs()
    training_settings.read(os.path.join(path_dir,const.TRAIN_INPUT_FILENAME))

    # with open(os.path.join(path_dir,'dataset_src_file.txt')) as f:
    #     path_pkl=f.readline().replace('\n','')
    #     path_pkl=f.readline().replace('\n','')
    
    path_pkl=training_settings.dataset_src_file[1]
    data_sel= training_settings.data_select
    # Data loading
    raw_data=get_XY_from_MalabDataSet(path=path_pkl, data_sel= data_sel,verbose=verbose)#, type2load='.pkl')
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

    training_settings=TrainInputs()
    training_settings.read(os.path.join(path_dir,const.TRAIN_INPUT_FILENAME))

    # with open(os.path.join(path_dir,'dataset_src_file.txt')) as f:
    #     path_pkl=f.readline().replace('\n','')
    #     path_pkl=f.readline().replace('\n','')
    
    path_pkl=training_settings.dataset_src_file[1]
    path_pkl=""
    data_sel= training_settings.data_select
    # Data loading
    raw_data=get_XY_from_MalabDataSet(path=path_pkl, data_sel= data_sel,verbose=verbose)#, type2load='.pkl')

    #idx_samples_path= training_settings.idx_samples_file[0]
    #idx_samples_path=""
    #tmp= MatlabDataSet(verbose=True)
    #tmp.mk_dataset_from_matlab(path= idx_samples_path, only_get_samples_EIDORS=True)
    #print(tmp.samples_EIDORS['idx_train'].flatten() ,tmp.samples_EIDORS['idx_train'].flatten().shape)
    #idx=[]
    #idx.append(f2(tmp.samples_EIDORS['idx_train'].flatten()))
    ##idx.append(f2(tmp.samples_EIDORS['idx_val'].flatten() ))
    #idx.append(f2(tmp.samples_EIDORS['idx_test'].flatten() ))
    #print(idx)



    eval_dataset = dataloader(raw_data, use_tf_dataset=False,verbose=verbose, train_inputs=training_settings)
    #save_idx_samples_2matfile(raw_data,eval_dataset,get_date_time())
    if verbose:
        print(eval_dataset.use_tf_dataset)
        if eval_dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in eval_dataset.test.as_numpy_iterator():
                print('samples size:',inputs.shape, outputs.shape)
                # Print the first element and the label
                # print(inputs[0])
                # print('label of this input is', outputs[0]
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
    path, filename= get_file(filetypes=[("Matlab files","*.mat*")], initialdir=os.path.split(training_settings.idx_samples_file[0])[0])
    #perm_eidors_path= os.path.join(os.path.split(training_settings.idx_samples_file[0])[0], 'GN/elems_solved_GN.mat')
    perm_eidors_path= os.path.join(path, filename)
    tmp= MatlabDataSet(verbose=True)
    if verify_file(perm_eidors_path, const.EXT_MAT):
        tmp.mk_dataset_from_matlab(path= perm_eidors_path, only_get_samples_EIDORS=True)
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
    results.append(error_eval(perm_real[:nb_eidors,:], perm_eidors[:nb_eidors,:], verbose=False, axis_samples=0, info='Results EIDORS'))
   
    #Generate 5 random numbers between 10 and 30
    #randnlist = random.sample(range(nb_eidors), 3)
    randnlist=[range(3)] 
    plot_real_NN_EIDORS(eval_dataset.fwd_model, perm_real[randnlist,:].T, perm_nn[randnlist,:].T, perm_eidors[randnlist,:].T)
    
    plot_eval_results(results, axis='linear')

def std_eidors_eval_pipeline_lazy_version(verbose=False):

    title= 'Select directory of model to evaluate'
    path_dir=get_dir(title=title)

    #path_dir='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Std_keras_20211007_144918'
    if not path_dir:
        return
    # read train inputs instead

    training_settings=TrainInputs()
    training_settings.read(os.path.join(path_dir,const.TRAIN_INPUT_FILENAME))

    # with open(os.path.join(path_dir,'dataset_src_file.txt')) as f:
    #     path_pkl=f.readline().replace('\n','')
    #     path_pkl=f.readline().replace('\n','')
    
    path_pkl=training_settings.dataset_src_file[1]
    data_sel= training_settings.data_select
    # Data loading
    raw_data=get_XY_from_MalabDataSet(path=path_pkl, data_sel= data_sel,verbose=verbose)#, type2load='.pkl')

    idx_samples_path= training_settings.idx_samples_file[0]
    tmp= MatlabDataSet(verbose=True)
    tmp.mk_dataset_from_matlab(path= idx_samples_path, only_get_samples_EIDORS=True)
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
                # print('label of this input is', outputs[0]
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
    path, filename= get_file(filetypes=[("Matlab files","*.mat*")], initialdir=os.path.split(training_settings.idx_samples_file[0])[0])
    #perm_eidors_path= os.path.join(os.path.split(training_settings.idx_samples_file[0])[0], 'GN/elems_solved_GN.mat')
    perm_eidors_path= os.path.join(path, filename)
    tmp= MatlabDataSet(verbose=True)
    if verify_file(perm_eidors_path, const.EXT_MAT):
        tmp.mk_dataset_from_matlab(path= perm_eidors_path, only_get_samples_EIDORS=True)
        perm_gn= tmp.samples_EIDORS['elem_data'].T # matlab samples are columnwise sorted
        perm_gn=scale_prepocess(perm_gn, True)
        print(perm_gn, perm_gn.shape)
        #perm_eidors_n= tmp.samples_EIDORS['elem_data_n'].T
    else:
        print('############ no data from EIDORS available ###############')
        perm_eidors= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])
        perm_eidors=tf.keras.utils.normalize(perm_eidors, axis=0).astype("float32")
        #perm_eidors_n= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])
 
    path, filename= get_file(filetypes=[("Matlab files","*.mat*")], initialdir=os.path.split(training_settings.idx_samples_file[0])[0])
    #perm_eidors_path= os.path.join(os.path.split(training_settings.idx_samples_file[0])[0], 'GN/elems_solved_GN.mat')
    perm_eidors_path= os.path.join(path, filename)
    tmp= MatlabDataSet(verbose=True)
    if verify_file(perm_eidors_path, const.EXT_MAT):
        tmp.mk_dataset_from_matlab(path= perm_eidors_path, only_get_samples_EIDORS=True)
        perm_tv= tmp.samples_EIDORS['elem_data'].T # matlab samples are columnwise sorted
        perm_tv=scale_prepocess(perm_tv, True)
        print(perm_tv, perm_tv.shape)
        #perm_eidors_n= tmp.samples_EIDORS['elem_data_n'].T
    else:
        print('############ no data from EIDORS available ###############')
        perm_eidors= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])
        perm_eidors=tf.keras.utils.normalize(perm_eidors, axis=0).astype("float32")
        #perm_eidors_n= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])
        
    nb_eidors= perm_gn.shape[0]

    results= list()
    results.append(error_eval(perm_real[:nb_eidors,:], perm_nn[:nb_eidors,:], verbose=False, axis_samples=0, info='Results ML'))
    results.append(error_eval(perm_real[:nb_eidors,:], perm_gn[:nb_eidors,:], verbose=False, axis_samples=0, info='Results GN'))
    results.append(error_eval(perm_real[:nb_eidors,:], perm_tv[:nb_eidors,:], verbose=False, axis_samples=0, info='Results TV'))
    #Generate 5 random numbers between 10 and 30
    randnlist = random.sample(range(nb_eidors), 3)
    labels= ['True','Results ML','Results GN','Results TV']
    # randnlist=[range(3)] 
    plot_real_NN_EIDORS(eval_dataset.fwd_model, labels, perm_real[randnlist,:].T, perm_nn[randnlist,:].T, perm_gn[randnlist,:].T, perm_tv[randnlist,:].T)
    
    plot_eval_results(results, axis='linear')
    
def std_several_ML_eval_pipline_lazy_version(verbose=False):
    
    nb_eidors=1000
    results= list()
    for i in range(9):
        title= 'Select directory of model to evaluate'
        path_dir=get_dir(title=title)

        #path_dir='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Std_keras_20211007_144918'
        if not path_dir:
            return
        # read train inputs instead

        training_settings=TrainInputs()
        training_settings.read(os.path.join(path_dir,const.TRAIN_INPUT_FILENAME))

        # with open(os.path.join(path_dir,'dataset_src_file.txt')) as f:
        #     path_pkl=f.readline().replace('\n','')
        #     path_pkl=f.readline().replace('\n','')
        
        path_pkl=training_settings.dataset_src_file[1]
        data_sel= training_settings.data_select
        # Data loading
        raw_data=get_XY_from_MalabDataSet(path=path_pkl, data_sel= data_sel,verbose=verbose)#, type2load='.pkl')

        idx_samples_path= training_settings.idx_samples_file[0]
        tmp= MatlabDataSet(verbose=True)
        tmp.mk_dataset_from_matlab(path= idx_samples_path, only_get_samples_EIDORS=True)
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
                    # print('label of this input is', outputs[0]
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
        perm_nn =gen.mk_prediction(eval_dataset.test,steps=steps)
        # model_evaluation= gen.model.evaluate(eval_dataset.test,steps=steps)

        # print('model_evaluation',model_evaluation)


        #  eval predictions vs
        
        results.append(error_eval(perm_real[:nb_eidors,:], perm_nn[:nb_eidors,:], verbose=False, axis_samples=0, info=str(i+1)))
        # results.append(error_eval(perm_real[:nb_eidors,:], perm_nn(1)[:nb_eidors,:], verbose=False, axis_samples=0, info='Results 1st best optML'))
        # results.append(error_eval(perm_real[:nb_eidors,:], perm_nn(2)[:nb_eidors,:], verbose=False, axis_samples=0, info='Results 2nd best optML'))
    #Generate 5 random numbers between 10 and 30
    # randnlist = random.sample(range(nb_eidors), 3)
    # labels= ['True','Results NN','Results GN','Results TV']
    # randnlist=[range(3)] 
    # plot_real_NN_EIDORS(eval_dataset.fwd_model, labels, perm_real[randnlist,:].T, perm_nn[0][randnlist,:].T, perm_nn[1][randnlist,:].T, perm_nn[2][randnlist,:].T)
    for i in range(3):
        plot_eval_one_results( i, results, axis='linear')
    

if __name__ == "__main__":

    # a= TrainInputs()
    # a.read('E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Std_keras_20211006_165901/train_inputs.txt')


    # path_pkl= 'datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    # print(verify_file(path_pkl, extension=".pkl", debug=True))
    # get_XY_from_MalabDataSet(path=path_pkl, data_sel= ['Xih','Yih'],verbose=True)
    std_eval_pipeline(verbose=True)
    #std_eidors_eval_pipeline(verbose=True)
    # std_eidors_eval_pipeline_lazy_version(verbose=False)
    #std_several_ML_eval_pipline_lazy_version(verbose=False)

    plt.show()
    
