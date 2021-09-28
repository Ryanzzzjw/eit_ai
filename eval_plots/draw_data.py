### this code is called by EVAL.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.io import loadmat
import tensorflow as tf
import autokeras as ak
import time

def data_loading(input_PATH, output_PATH, trianglesDATA_fPATH, model_NAME):
    start_time = time.time()
    trianglesDATA = loadmat(trianglesDATA_fPATH)
    inputDATA = loadmat(input_PATH)
    outputDATA = loadmat(output_PATH)
    model = tf.keras.models.load_model(model_NAME, custom_objects=ak.CUSTOM_OBJECTS)
    print("Data loading time = ", time.time() - start_time, 's')
    return inputDATA, outputDATA, trianglesDATA, model

def Extract_tr_data(trianglesDATA):
    
    # Load nodes coordinates data and triangle nodes
    triangles = np.array(trianglesDATA["tr_points"])
    xy = np.array(trianglesDATA["nodes"])
    x, y = xy.T 
    
    return x, y, triangles

def convert(conduct_elem, triangles, x):

    # Convert from each triangle element conductivity data to each node conductivity data.
    m = np.array(conduct_elem).flatten()
    n = np.array(triangles, dtype = float).flatten()
    z = [0 for a in range(np.size(x))]

    list = []

    for i in range (len(n)):
        idx = n[int(i)] 
        idx =int(idx)
        #print(idx)
        if idx not in list: 
            j= int(i/3)
            z[idx-1]=m[j]
            list.append(idx)
    
    return z


def plot_mesh(trianglesDATA, conduct_elem):
    
    # Load nodes coordinates data and triangle nodes
    print('Extracting mesh triangular data..')  
    x, y, triangles = Extract_tr_data(trianglesDATA)
    # Create triangulation.
    triang = mtri.Triangulation(x, y)
    z = convert(conduct_elem, triangles, x)
    
    # Plot the triangulation.
    print('Ploting results..')
    plt.plot()
    plt.tricontourf(triang, z)
    plt.triplot(triang, '-', alpha=.5)
    plt.tight_layout()
    tpc = plt.tripcolor(triang, z, shading='flat')
    clb = plt.colorbar(tpc)
    if z[0] <= 1:
        clb.ax.set_ylabel('Normalized conductivity distribution')
    else:
        clb.ax.set_ylabel('Conductivity distribution')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()


def draw_MLsolver(inputDATA, model, trianglesDATA, meas_index = 0):

    global start_time
    start_time = time.time()

    # Load measurement data
    Xih = inputDATA
    Xih = Xih["Xih"].T
    print('Input size: ' + str(Xih[0].size) )
    
    print('Solving model..')
    # Norm data
    x_train = tf.keras.utils.normalize(Xih, axis=1)

    # Solve ML model 
    voltage =  np.expand_dims(x_train[meas_index], 0) 
    conduct_elem = model.predict(voltage)
    print("Solving time = ", time.time() - start_time, 's')

    plot_mesh(trianglesDATA, conduct_elem)

def draw_EIDORS_data(outputDATA, trianglesDATA, meas_index = 0):

    global start_time
    start_time = time.time()
    print('Loading output data..')
    Yih = outputDATA
    Yih = Yih["Yih"].T
    print('Output size: ' + str(Yih[0].size) )
    conduct_elem = Yih[meas_index]

    plot_mesh(trianglesDATA, conduct_elem)

'''
def draw_compare(input_PATH, output_PATH, trianglesDATA_fPATH, model_NAME, meas_index = 0):
    
    # Load measurement data
    Xih = loadmat(input_PATH)
    Xih = Xih["Xih"].T
    Yih = loadmat(output_PATH)
    Yih = Yih["Yih"].T
    print('Input size: ' + Xih[0].size, + 'Output size: ' +Yih[0].size)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax = axs.ravel()  

    axs[0].tricontourf(triang, z0)
    axs[0].triplot(triang, '-', alpha=.5)
    axs[0].set_title('FWD model')

    tpc0 = axs[0].tripcolor(triang, z0, shading='flat')
    fig.colorbar(tpc0, ax = ax[0])

    axs[1].tricontourf(triang, z)
    axs[1].triplot(triang, '-', alpha=.5)
    axs[1].set_title('Model after ML')
    tpc = axs[1].tripcolor(triang, z, shading='flat')
    fig.colorbar(tpc, ax = ax[1])
    fig.tight_layout()
    plt.show()
'''