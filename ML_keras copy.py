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


# tf.compat.v1.ConfigProto()

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.333
# session = tf.compat.v1.InteractiveSession(config=config)

def keras_model():
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input

    inputs = Input(shape=(256,))
    x = Dense(512, activation=tf.nn.relu)(inputs)
    x = Dense(512, activation=tf.nn.relu)(x)
    outputs = Dense(990, activation=tf.nn.sigmoid)(x)

    return keras.Model(inputs, outputs)

def get_compiled_model():

    model= keras_model()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.Accuracy()],
    )
    return model


def get_dataset(batch_size = 32, num_val_samples = 5000, test_size= 0.20, val_size=0.20):

    if test_size+val_size>=0.8:
        test_size= 0.2
        val_size=0.2


    # Load data
    Xih = loadmat(r"Xih_990x50k.mat") # <class 'numpy.ndarray'> 
    Xih = Xih["Xih"].T

    Yih = loadmat(r"Yih_990x50k.mat")
    Yih = Yih["Yih"].T

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

def training_pipeline():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.OneDeviceStrategy(gpus[0])

    
    #print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    train_dataset, val_dataset, test_dataset = get_dataset()

    for inputs, outputs in train_dataset.as_numpy_iterator():
        # Verify the shapes are still as we expect
        print("Input shape is:", inputs.shape, "output shape is:", outputs.shape)

        # Print the first element and the label
        print(inputs[0,:])
    
        print('label of this input is', outputs[0])
        
        # Break now. We only want to visualise the first example
        break

    # a= train_dataset.take(1)
    # print('hhhhhhhhhhhhhhhhhhhhhhh',len(list(train_dataset)),len(list(val_dataset)),len(list(test_dataset)))
    # iterator = val_dataset.__iter__()
    # next_element = iterator.get_next()
    # pt = next_element[0]
    # en = next_element[1]
    # print(np.shape(pt.numpy()))
    # print(np.shape(en.numpy()))

    # Open a strtensor flowategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = get_compiled_model()
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
