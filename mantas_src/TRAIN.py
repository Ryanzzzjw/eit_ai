##### For training #####

### after running the code, logs file should be created automatically; 
### using Tensorboard: open CMD, where logs file or this code is, and type " tensorboard --logdir=logs/ " or " py -m tensorboard.main --logdir=logs/ " or etc.
### then it can be seen http://H-PC:600 or http://localhost:6006/ or etc., copy it and paste it online to see training graphs live or already executed
### CMD must be opened all the time, when is wanted to see graphs. 

from train.training_models import ML_autoKERAS, ML_KERAS, training_data_loading, ML_optimization_TB
import os

PATH = "C:\\Users\manta\Documents\Master thesis\Tr" #path to train files 

#choose one data set:
if 1:   #for 3054 elements (max mesh el. size od 0.05)
    input_fNAME = 'Xih_3054x50k.mat'
    output_fNAME = 'Yih_3054x50k.mat'

if 0:   #for 990 elements  (max mesh el. size od 0.09)
    input_fNAME = 'Xih_990x50k.mat'
    output_fNAME = 'Yih_990x50k.mat'    

input_PATH = os.sep.join([input_fNAME])
output_PATH = os.sep.join([output_fNAME])


inputDATA, outputDATA = training_data_loading(input_PATH, output_PATH)  

#choose one:   (last one will take the most time, because different models will be trained, but it's important now!)
if 0:   # 1 to train randomly made Tf-Keras model 
    nameMODEL = 'test_ML_Keras' 
    ML_KERAS(inputDATA, outputDATA, nameMODEL)
if 0:   # 1 to train autoKeras model   
    nameMODEL = 'test_ML_autoKeras'
    ML_autoKERAS(inputDATA, outputDATA, nameMODEL)

if 1:   # 1 to train several Tf-Keras models with given NN parameters 
    dense_layers = [0,1,2]
    layer_sizes = [128, 256, 512, 1028]
    ML_optimization_TB(inputDATA, outputDATA, dense_layers, layer_sizes) 
