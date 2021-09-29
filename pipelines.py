import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.io import loadmat
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.model_selection
import time

from tensorflow.python.keras.metrics import FalseNegatives
from data_preprocessing.load_mat_files import *
from data_preprocessing.dataset import *
from model.train_models import *

from datetime import datetime

def std_training_pipeline(verbose=False):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.OneDeviceStrategy(gpus[0])

    
    #print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Data loading
    training_dataset = dataloader()
    
    if verbose:
        # extract data for verification?
        for inputs, outputs in training_dataset.train.as_numpy_iterator():
            print(inputs.size, outputs.size)
            # Print the first element and the label
            print(inputs[0,:])
            print('label of this input is', outputs[0])
            break

    # Model setting

    EPOCH= 100
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = training_dataset.nb_samples // BATCH_SIZE
    LEARNING_RATE= 0.1
    OPTIMIZER=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    LOSS=keras.losses.CategoricalCrossentropy()
    METRICS=[keras.metrics.Accuracy()]

    gen = ModelGenerator()
    gen.select_model(model_func=gen.std_keras,
                    input_size=training_dataset.features_size,
                    output_size=training_dataset.labels_size)
    gen.compile_model(OPTIMIZER, LOSS, METRICS)
    print(gen.model.summary())

    start_time = time.time()
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    NAME = "Model_{}_{}".format(gen.name,  date_time )

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    # Train the model on all available devices.
    gen.mk_fit(training_dataset,
                epochs=3,
                callbacks=[tensorboard],
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=STEPS_PER_EPOCH)
    gen.save_model()             
    
    # save model

    # Test the model on all available devices.
   # model.evaluate(test_dataset)





if __name__ == "__main__":
    # import subprocess
    # cmd = [ 'tensorboard', '--logdir=logs/' ]
    # output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
    # print(output)

    
    std_training_pipeline(verbose=True)
