

from sys import modules
from numpy.core.fromnumeric import reshape
from scipy.io import loadmat

from tensorflow.python.keras.metrics import FalseNegatives
from modules.load_mat_files import *
from modules.dataset import *
from modules.train_models import *
from modules.draw_data import *
from modules.path_utils import mk_ouput_dir, get_dir, verify_file
from modules.eval_utils import error_eval
import numpy as np
import matplotlib.pyplot as plt


def std_eval_pipeline(verbose=False):


    title= 'Select directory of model to evaluate'
    # path_dir=get_dir(title=title)
    path_dir='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/outputs/Model_std_keras_20211004_170742'
    
    
    with open(os.path.join(path_dir,'training_dataset_src_file.txt')) as f:
        path_pkl=f.readline()
    
    # Data loading
    raw_data=get_XY_from_MalabDataSet(path=path_pkl, data_sel= ['Xih','Yih'],verbose=verbose)#, type2load='.pkl')
    training_dataset = dataloader(raw_data, use_tf_dataset=True,verbose=verbose)
    
    if verbose:
        print(training_dataset.use_tf_dataset)
        if training_dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in training_dataset.test.as_numpy_iterator():
                print(inputs.shape, outputs.shape)
                # Print the first element and the label
                # print(inputs[0])
                # print('label of this input is', outputs[0])
                # plot_EIT_samples(training_dataset.fwd_model, outputs[0], inputs[0])
                break
        save_2matfile_4_solving(raw_data,training_dataset)        

    # Load model
    gen = ModelGenerator()
    gen.load_model(os.path.join(path_dir,'model.model'))
    print(gen.model.summary())

    # make predictions
    print('steps', training_dataset.test_len//32)
    results= gen.model.evaluate(training_dataset.test,steps=training_dataset.test_len//32)

    print('loss accuracy',results)
    
    perm_nn = gen.model.predict(training_dataset.test,steps=training_dataset.test_len//32)

    print('perm_NN', perm_nn.shape)
    # plot_EIT_samples(training_dataset.fwd_model, perm_NN[0].T, inputs[0])

    #  eval predictions vs
    # load Eidors samples
    path, _ = os.path.split(path_pkl)
    path= os.path.join(path, 'elems_solved.mat')
    a= MatlabDataSet(verbose=True)
    a.mk_dataset_from_matlab(path= verify_file(path=path, extension='.mat'), only_get_samples_EIDORS=True)
    for key in a.samples_EIDORS.keys():
        print('samples_EIDORS[{}]'.format(key) , a.samples_EIDORS[key].shape)
    perm_eidors= a.samples_EIDORS['elem_data'].T
    perm_eidors_n= a.samples_EIDORS['elem_data_n'].T

    print('perm_eidors', perm_eidors.shape)
    print('perm_eidors_n',perm_eidors_n.shape)
    # get real perm

    l=[]
    for i, xy in enumerate(training_dataset.test):
        # print('#', i, training_dataset.batch_size,training_dataset.test_len) 
        if (i+1)*training_dataset.batch_size>training_dataset.test_len:
            break
        l.append(xy[1].numpy())
            
    perm_real = np.concatenate(l, axis=0)
    print('perm_real',perm_real.shape)




    plot_real_NN_EIDORS(training_dataset.fwd_model, perm_real[0,:].T, perm_nn[0,:].T, perm_eidors[0,:].T)




    mse_nn, rie_nn, icc_nn= error_eval(perm_real, perm_nn[:perm_real.shape[0],:], verbose=False, axis_samples=0)

    mse_eidors, rie_eidors, icc_eidors= error_eval(perm_real,perm_eidors[:perm_real.shape[0],:], verbose=False, axis_samples=0)

    

    fig1, ax = plt.subplots(1,3)
    ax[0].set_title('MSE')
    ax[0].boxplot((mse_nn, mse_eidors))
    ax[1].set_title('RIE')
    ax[1].boxplot((rie_nn, rie_eidors))
    ax[2].set_title('icc')
    ax[2].boxplot((icc_nn, icc_eidors))
    plt.show()





    


    # plot some samples



    # # Model setting

    # EPOCH= 10
    # BATCH_SIZE = 32
    # STEPS_PER_EPOCH = training_dataset.train_len // BATCH_SIZE
    # VALIDATION_STEP = training_dataset.val_len // BATCH_SIZE
    # LEARNING_RATE= 0.1
    # OPTIMIZER=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # LOSS='binary_crossentropy' #keras.losses.CategoricalCrossentropy()
    # METRICS=[keras.metrics.Accuracy()]

    # gen = ModelGenerator()
    # gen.std_keras(input_size=training_dataset.features_size,
    #                 output_size=training_dataset.labels_size)
    # gen.compile_model(OPTIMIZER, LOSS, METRICS)

    # now = datetime.now()
    # date_time = now.strftime("%Y%m%d_%H%M%S")
    # NAME = "Model_{}_{}".format(gen.name,  date_time )
    # ouput_dir= mk_ouput_dir(NAME)
    # tensorboard = TensorBoard(log_dir= os.path.join(ouput_dir,'tf_boards_logs'))
    # log_tensorboard(os.path.join(ouput_dir,'tf_boards_logs'))

    # # Train the model on all available devices.
    # gen.mk_fit(training_dataset,
    #             epochs=EPOCH,
    #             callbacks=[tensorboard],
    #             steps_per_epoch=STEPS_PER_EPOCH,
    #             validation_steps=VALIDATION_STEP)
    # gen.save_model(path=ouput_dir)             
    
    # save model

    # Test the model on all available devices.
   # model.evaluate(test_dataset)

if __name__ == "__main__":

    std_eval_pipeline(verbose=True)
    
