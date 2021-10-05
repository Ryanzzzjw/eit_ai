

import tensorflow.keras as keras


from modules.load_mat_files import *
from modules.dataset import *
from modules.train_models import *
from modules.draw_data import *


from datetime import datetime
from tensorboard import program

from modules.path_utils import mk_ouput_dir


def log_tensorboard(log_path):

    tracking_address = log_path # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"\n######################################\nTensorflow listening on {url}\n######################################\n")



def std_training_pipeline(verbose=False):

    # Data loading
    path=''# 'E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose)
    training_dataset = dataloader(raw_data, use_tf_dataset=True,verbose=verbose)
    
    if verbose:
        if training_dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in training_dataset.train.as_numpy_iterator():
                # print(inputs.size, outputs.size)
                # Print the first element and the label
                print(inputs[0,:])
                print('label of this input is', outputs[0])
                plot_EIT_mesh(training_dataset.fwd_model, outputs[0])
                break

    # Model setting

    EPOCH= 10
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = training_dataset.train_len // BATCH_SIZE
    VALIDATION_STEP = training_dataset.val_len // BATCH_SIZE
    LEARNING_RATE= 0.1
    OPTIMIZER=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    LOSS='binary_crossentropy' #keras.losses.CategoricalCrossentropy()
    METRICS=[keras.metrics.Accuracy()]

    gen = ModelGenerator()
    gen.std_keras(input_size=training_dataset.features_size,
                    output_size=training_dataset.labels_size)
    gen.compile_model(OPTIMIZER, LOSS, METRICS)

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    NAME = "Model_{}_{}".format(gen.name,  date_time )
    ouput_dir= mk_ouput_dir(NAME)

    with open(os.path.join(ouput_dir,'training_dataset_src_file.txt'), 'w') as f:
        f.write(training_dataset.src_file)
    with open(os.path.join(ouput_dir,'training_dataset_src_file.txt')) as f:
        print(f.readlines())

    tensorboard = TensorBoard(log_dir= os.path.join(ouput_dir,'tf_boards_logs'))
    log_tensorboard(os.path.join(ouput_dir,'tf_boards_logs'))

    # Train the model on all available devices.
    gen.mk_fit(training_dataset,
                epochs=EPOCH,
                callbacks=[tensorboard],
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=VALIDATION_STEP)
    gen.save_model(path=ouput_dir)             
    
    # save model

    # Test the model on all available devices.
   # model.evaluate(test_dataset)


def std_auto_pipeline(verbose=False):
    
    # Data loading
    path= 'E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    raw_data=get_XY_from_MalabDataSet(path=path, data_sel= ['Xih','Yih'],verbose=verbose)
    training_dataset = dataloader(raw_data, use_tf_dataset=False,verbose=verbose)
    
    if verbose:
        if training_dataset.use_tf_dataset:
            # extract data for verification?
            for inputs, outputs in training_dataset.train.as_numpy_iterator():
                # print(inputs.size, outputs.size)
                # Print the first element and the label
                print(inputs[0,:])
                print('label of this input is', outputs[0])
                plot_EIT_mesh(training_dataset.fwd_model, outputs[0])
                break

    # Model setting
    EPOCH= 2
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = training_dataset.train_len // BATCH_SIZE
    VALIDATION_STEP = training_dataset.val_len // BATCH_SIZE
    LEARNING_RATE= 0.1
    OPTIMIZER=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    LOSS='binary_crossentropy' #keras.losses.CategoricalCrossentropy()
    METRICS=[keras.metrics.Accuracy()]

    gen = ModelGenerator()
    gen.std_autokeras(input_size=training_dataset.features_size,
                    output_size=training_dataset.labels_size,max_trials=2)

    gen.compile_model(OPTIMIZER, LOSS, METRICS)

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    NAME = "Model_{}_{}".format(gen.name,  date_time )
    ouput_dir= mk_ouput_dir(NAME)

    with open(os.path.join(ouput_dir,'training_dataset_src_file.txt'), 'w') as f:
        f.write(training_dataset.src_file)
    with open(os.path.join(ouput_dir,'training_dataset_src_file.txt')) as f:
        print(f.readlines())


    tensorboard = TensorBoard(log_dir= os.path.join(ouput_dir,'tf_boards_logs'))
    log_tensorboard(os.path.join(ouput_dir,'tf_boards_logs'))

    # Train the model on all available devices.
    gen.mk_fit(training_dataset,
                epochs=EPOCH,
                callbacks=[tensorboard],
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=VALIDATION_STEP)
    #Save model
    gen.save_model(path=ouput_dir)          

if __name__ == "__main__":

    #std_training_pipeline(verbose=True)
    std_auto_pipeline(verbose=True)
