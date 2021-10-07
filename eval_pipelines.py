

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


def std_eval_pipeline(verbose=False):


    title= 'Select directory of model to evaluate'
    path_dir=get_dir(title=title)
    #path_dir='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Model_std_keras_20211004_170742'
    
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
    eval_dataset = dataloader(raw_data, use_tf_dataset=True,verbose=verbose, train_inputs=training_settings)
    
    if verbose:
        print(eval_dataset.use_tf_dataset)
        if eval_dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in eval_dataset.test.as_numpy_iterator():
                print(inputs.shape, outputs.shape)
                # Print the first element and the label
                # print(inputs[0])
                # print('label of this input is', outputs[0])
                plot_EIT_samples(eval_dataset.fwd_model, outputs[0], inputs[0])
                break
    

    
    # Extract True values of conductivity

    # l=[]
    # for i, xy in enumerate(eval_dataset.test):
    #     # print('#', i, eval_dataset.batch_size,eval_dataset.test_len) 
    #     if (i+1)*eval_dataset.batch_size>eval_dataset.test_len:
    #         break
    #     l.append(xy[1].numpy())
            
    # perm_real = np.concatenate(l, axis=0)

    perm_real=extract_samples(eval_dataset, dataset_part='test', idx_samples=None, elem_idx = 1)

    print('\nperm_real',perm_real.shape)

    # Load model
    gen = ModelGenerator()
    gen.load_model(training_settings.model_saving_path)
    print(gen.model.summary())

    # make predictions
    steps = eval_dataset.test_len//eval_dataset.batch_size
    model_evaluation= gen.model.evaluate(eval_dataset.test,steps=steps)

    print('model_evaluation',model_evaluation)
    
    perm_nn = gen.model.predict(eval_dataset.test,steps=steps)

    print('perm_NN', perm_nn.shape)
    # plot_EIT_samples(eval_dataset.fwd_model, perm_NN[0].T, inputs[0])

    #  eval predictions vs
    # load Eidors samples
    perm_eidors_path= training_settings.idx_samples_file[0].replace(const.EXT_IDX_FILE, const.EXT_EIDORS_SOLVING_FILE)
    tmp= MatlabDataSet(verbose=True)
    if verify_file(perm_eidors_path, const.EXT_MAT):
        tmp.mk_dataset_from_matlab(path= perm_eidors_path, only_get_samples_EIDORS=True)
        perm_eidors= tmp.samples_EIDORS['elem_data'].T # matlab samples are columnwise sorted
        perm_eidors_n= tmp.samples_EIDORS['elem_data_n'].T

    else:
        print('############ no data from EIDORS available ###############')
        perm_eidors= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])
        perm_eidors=tf.keras.utils.normalize(perm_eidors, axis=0).astype("float32")
        perm_eidors_n= np.random.randn(perm_nn.shape[0],perm_nn.shape[1])

    
    results= list()
    results.append(error_eval(perm_real, perm_nn[:perm_real.shape[0],:], verbose=False, axis_samples=0, info='Results NN'))
    results.append(error_eval(perm_real, perm_eidors[:perm_real.shape[0],:], verbose=False, axis_samples=0, info='Results Eidors'))

    plot_real_NN_EIDORS(eval_dataset.fwd_model, perm_real[2,:].T, perm_nn[2,:].T, perm_eidors[2,:].T)
    

    
    plot_eval_results(results, axis='linear')

    # plot some samples

    # # Model setting

    # EPOCH= 10
    # BATCH_SIZE = 32
    # STEPS_PER_EPOCH = eval_dataset.train_len // BATCH_SIZE
    # VALIDATION_STEP = eval_dataset.val_len // BATCH_SIZE
    # LEARNING_RATE= 0.1
    # OPTIMIZER=keras.optimizers.Adam(ling_rate=LEARNING_RATE)
    # LOSS='binary_crossentropy' #keras.losses.CategoricalCrossentropy()
    # METRICS=[keras.metrics.Accuracy()]

    # gen = ModelGenerator()
    # gen.std_keras(input_size=eval_dataset.features_size,
    #                 output_size=eval_dataset.labels_size)
    # gen.compile_model(OPTIMIZER, LOSS, METRICS)

    # now = datetime.now()
    # date_time = now.strftime("%Y%m%d_%H%M%S")
    # NAME = "Model_{}_{}".format(gen.name,  date_time )
    # ouput_dir= mk_ouput_dir(NAME)
    # tensorboard = TensorBoard(log_dir= os.path.join(ouput_dir,'tf_boards_logs'))
    # log_tensorboard(os.path.join(ouput_dir,'tf_boards_logs'))

    # # Train the model on all available devices.
    # gen.mk_fit(eval_dataset,
    #             epochs=EPOCH,
    #             callbacks=[tensorboard],
    #             steps_per_epoch=STEPS_PER_EPOCH,
    #             validation_steps=VALIDATION_STEP)
    # gen.save_model(path=ouput_dir)             
    
    # save model

    # Test the model on all available devices.
   # model.evaluate(test_dataset)

if __name__ == "__main__":

    # a= TrainInputs()
    # a.read('E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Std_keras_20211006_165901/train_inputs.txt')


    # path_pkl= 'datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    # print(verify_file(path_pkl, extension=".pkl", debug=True))
    # get_XY_from_MalabDataSet(path=path_pkl, data_sel= ['Xih','Yih'],verbose=True)
    std_eval_pipeline(verbose=True)

    plt.show()
    
