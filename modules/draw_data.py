### this code is called by EVAL.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.io import loadmat

import modules.interp2d as interp2d

from modules.utils import check_order

# def data_loading(input_PATH, output_PATH, trianglesDATA_fPATH, model_NAME):
#     start_time = time.time()
#     trianglesDATA = loadmat(trianglesDATA_fPATH)
#     inputDATA = loadmat(input_PATH)
#     outputDATA = loadmat(output_PATH)
#     model = tf.keras.models.load_model(model_NAME, custom_objects=ak.CUSTOM_OBJECTS)
#     print("Data loading time = ", time.time() - start_time, 's')
#     return inputDATA, outputDATA, trianglesDATA, model

# def Extract_tr_data(fwd_model):
    
#     # Load nodes coordinates data and triangle nodes
#     triangles = np.array(fwd_model['elems'])
#     xy = np.array(fwd_model['nodes'])
#     x, y = xy.T 
    
#     return x, y, triangles

# def convert(conduct_elem, triangles, x):

#     # Convert from each triangle element conductivity data to each node conductivity data.
#     m = np.array(conduct_elem).flatten()
#     n = np.array(triangles, dtype = float).flatten()
#     z = [0 for a in range(np.size(x))]

#     list = []

#     for i in range (len(n)):
#         idx = n[int(i)] 
#         idx =int(idx)
#         #print(idx)
#         if idx not in list: 
#             j= int(i/3)
#             z[idx-1]=m[j]
#             list.append(idx)
#     return z

def get_elem_nodal_data(fwd_model, perm):

    tri = np.array(fwd_model['elems'])
    pts = np.array(fwd_model['nodes'])

    
    # perm= fwd_model['un2']    
    perm= np.reshape(perm, (perm.shape[0],))

    tri = tri-1 # matlab count from 1 python from 0
    tri= check_order(pts, tri)
    data=dict()

    if perm.shape[0]==pts.shape[0]:
        data['elems_data'] = interp2d.pts2sim(tri, perm)
        data['nodes_data']= perm
    elif perm.shape[0]==tri.shape[0]:
        data['elems_data'] = perm
        data['nodes_data']= interp2d.sim2pts(pts, tri, perm)

    for key in data.keys():
        data[key]= np.reshape(data[key], (data[key].shape[0],))

    return tri, pts, data

def plot_EIT_mesh(fwd_model, perm):
    """[summary]

    Args:
        fwd_model ([type]): 
        perm ([type]): can be nodal_data or elem_data
    """
    
    tri, pts, data= get_elem_nodal_data(fwd_model, perm)


    fig, ax = plt.subplots(1,2)
    for i, key in enumerate(data.keys()):

        im = ax[i].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[key]),shading='flat', vmin=None,vmax=None)
        title= key

        if np.all(perm <= 1):
            title= title +'\nNormalized conductivity distribution'
        else:
            title= title +'\nConductivity distribution'
        ax[i].set_title(title)
        ax[i].set_xlabel("X axis")
        ax[i].set_ylabel("Y axis")
          
        ax[i].axis("equal")
        fig.colorbar(im,ax=ax[i])
    plt.show()

def plot_EIT_samples(fwd_model, perm, U):

    tri, pts, data= get_elem_nodal_data(fwd_model, perm)

    if perm.shape[0]==pts.shape[0]:
        key= 'nodes_data'
    else:
        key= 'elems_data'

    fig, ax = plt.subplots(1,2)
    

    im = ax[0].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[key]),shading='flat', vmin=None,vmax=None)
    title= key

    if np.all(perm <= 1):
        title= title +'\nNormalized conductivity distribution'
    else:
        title= title +'\nConductivity distribution'
    ax[0].set_title(title)
    ax[0].set_xlabel("X axis")
    ax[0].set_ylabel("Y axis")
        
    ax[0].axis("equal")
    fig.colorbar(im,ax=ax[0])

    ax[1].plot(U)
    plt.show()

def plot_real_NN_EIDORS(fwd_model, perm_real, perm_nn, perm_eidors):

    data= [dict() for _ in range(3)]

    tri, pts, data[0]= get_elem_nodal_data(fwd_model, perm_real)
    _, _, data[1]= get_elem_nodal_data(fwd_model, perm_nn)
    _, _, data[2]= get_elem_nodal_data(fwd_model, perm_eidors)

    # if perm.shape[0]==pts.shape[0]:
    #     key= 'nodes_data'
    # else:
    key= 'elems_data'

    fig, ax = plt.subplots(1,3)
    
    for i in range(3):
        im = ax[i].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[i][key]),shading='flat', vmin=None,vmax=None)
        title= key

        # if np.all(perm <= 1):
        #     title= title +'\nNormalized conductivity distribution'
        # else:
        #     title= title +'\nConductivity distribution'
        ax[i].set_title(title)
        ax[i].set_xlabel("X axis")
        ax[i].set_ylabel("Y axis")
            
        ax[i].axis("equal")
        fig.colorbar(im,ax=ax[i])

    plt.show()
    


    

if __name__ == "__main__":

    fmdl= loadmat('E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/test_plot.mat')
    plot_EIT_mesh(fmdl, fmdl['un2'])
