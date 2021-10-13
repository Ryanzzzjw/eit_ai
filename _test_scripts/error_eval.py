import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf


def error_eval(real_output, solved_output):

    #!!!I don't know why, but only in that form it normalized correctly, and if I don't put values in [], drops an error: Found input variables with inconsistent numbers of samples: [3054, 1]
    #real_output = tf.keras.utils.normalize(real_output, axis=1) #if I normalize like this, then it is normalized too much (like to 0.02.., not 1), maybe because of [3054, 1]
    real_output = [(real_output-min(real_output))/(max(real_output)-min(real_output))] #normalize 

    print('Real normalized values: ' + str(real_output) + '; Solved normalized values: ' + str(solved_output))

    mse = mean_squared_error(real_output, solved_output) #Mean Squared Error (MSE)
    print('MSE = ' + str(mse))

    rie = np.linalg.norm(real_output-solved_output)/np.linalg.norm(real_output) #Relative (Image) Error (RIE)
    print('RIE = ' + str(rie))

    icc = np.corrcoef(real_output, solved_output) #(Image) Correlation Coefficient (ICC)
    print('ICC = '+ str(icc[0, 1]))