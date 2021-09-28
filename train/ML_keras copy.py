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
from data_preprocessing.load_mat_files import *

# tf.compat.v1.ConfigProto()

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.333
def get_dataset(batch_size = 32, num_val_samples = 5000, test_size= 0.20, val_size=0.20):

    if test_size+val_size>=0.8:
        test_size= 0.2
        val_size=0.2

    # Load data

    b=MatlabDataSet()
    b= b.load_dataset_from_pickle()

    # Xih = loadmat(r".\\datasets\\Xih_990x50k.mat") # <class 'numpy.ndarray'> 
    Xih = b.samples["X"][:,:,1].T

    # Yih = loadmat(r".\\datasets\\Yih_990x50k.mat")
    Yih = b.samples["y"][:,:,1].T

    print(np.shape(Xih) ,np.shape(Yih))

    X = tf.keras.utils.normalize(Xih, axis=0).astype("float32")
    Y = Yih.astype("float32")
    
    # make the 
    
    N_samples= len(X)

    SAMPLES_dataset= tf.data.Dataset.from_tensor_slices((X, Y))

    TRAIN_dataset= SAMPLES_dataset.take(int((1-test_size+val_size)*N_samples))
    TEST_dataset= SAMPLES_dataset.take(int(test_size*N_samples))
    VAL_dataset= SAMPLES_dataset.take(int(val_size*N_samples))

    TRAIN_dataset= TRAIN_dataset.repeat().batch(batch_size)
    TEST_dataset= TEST_dataset.repeat().batch(batch_size)
    VAL_dataset=VAL_dataset.repeat().batch(batch_size)

    # Reserve num_val_samples samples for validation
  
    return (TRAIN_dataset, VAL_dataset, TEST_dataset)

# session = tf.compat.v1.InteractiveSession(config=config)

def keras_model(input_size=256, output_size=990):
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input

    inputs = Input(shape=(input_size,))
    x = Dense(512, activation=tf.nn.relu)(inputs)
    x = Dense(512, activation=tf.nn.relu)(x)
    outputs = Dense(output_size, activation=tf.nn.sigmoid)(x)

    return keras.Model(inputs, outputs)

def get_compiled_model(input_size=256, output_size=990):

    model= keras_model(input_size=input_size, output_size=output_size)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.Accuracy()],
    )
    return model




def training_pipeline():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.OneDeviceStrategy(gpus[0])

    
    #print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    train_dataset, val_dataset, test_dataset = get_dataset()

    for inputs, outputs in train_dataset.as_numpy_iterator():
        # Verify the shapes are still as we expect
        input_size= inputs.shape[1]
        output_size= outputs.shape[1]
        print("Input shape is:", inputs.shape[1], "output shape is:", outputs.shape[1])

        # Print the first element and the label
        print(inputs[0,:])
    
        print('label of this input is', outputs[0])
        
        # Break now. We only want to visualise the first example
        break

    # Open a strtensor flowategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = get_compiled_model(input_size=input_size, output_size=output_size)
    model.summary()

    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 50000 // BATCH_SIZE
    # Train the model on all available devices.
    t0=time.time()
    model.fit(train_dataset, epochs=10, validation_data=val_dataset, steps_per_epoch=(50000*0.6)//BATCH_SIZE, validation_steps= (50000*0.2)//BATCH_SIZE)
    print('Training lasted:',time.time()-t0 , 's')
    # Test the model on all available devices.
   # model.evaluate(test_dataset)

if __name__ == "__main__":
    training_pipeline()
