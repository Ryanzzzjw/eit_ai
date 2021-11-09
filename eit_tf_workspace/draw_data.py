### this code is called by EVAL.py

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyeit.eit.interp2d
import pyeit.mesh.utils
from scipy.io import loadmat

from eit_tf_workspace.eval_utils import *
from eit_tf_workspace.eval_utils import EvalResults




def get_elem_nodal_data(fwd_model, perm):

    tri = np.array(fwd_model['elems'])
    pts = np.array(fwd_model['nodes'])
    
    # perm= fwd_model['un2']    
    perm= np.reshape(perm, (perm.shape[0],))

    # tri = tri-1 # matlab count from 1 python from 0
    tri= pyeit.mesh.utils.check_order(pts, tri)
    data=dict()

    if perm.shape[0]==pts.shape[0]:
        data['elems_data'] = pyeit.eit.interp2d.pts2sim(tri, perm)
        data['nodes_data']= perm
    elif perm.shape[0]==tri.shape[0]:
        data['elems_data'] = perm
        data['nodes_data']= pyeit.eit.interp2d.sim2pts(pts, tri, perm)

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
    plt.show(block=False)

def format_inputs(fwd_model, data):
    if data.ndim==2:
        tri = np.array(fwd_model['elems'])
        pts = np.array(fwd_model['nodes'])
        if data.shape[1]==pts.shape[0] or data.shape[1]==tri.shape[0]:
            data= data.T

    return data

def plot_EIT_samples(fwd_model, perm, U):

    perm=format_inputs(fwd_model, perm)
    U=format_inputs(fwd_model, U)

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

    ax[1].plot(U.T)
    
    plt.show(block=False)

def plot_real_NN_EIDORS(fwd_model, perm_real,*argv):

    _perm= list()
    _perm.append(perm_real)

    for arg in argv:
        if _perm[0].shape==arg.shape:
            _perm.append(arg)

    perm=list()
    if perm_real.ndim > 1:
        n_row=  perm_real.shape[1]
        for p in _perm:
            perm.append(p)
    else:
        for p in _perm:
            perm.append(p.reshape((p.shape[0],1)))
        n_row= 1
    n_col = len(perm)

    fig, ax = plt.subplots(n_row,n_col)
    if ax.ndim==1:
        ax=ax.reshape((ax.shape[0],1)).T

    for row in range(ax.shape[0]):

        data= [dict() for _ in range(n_col)]
        for i, p in enumerate(perm):
            tri, pts, data[i]= get_elem_nodal_data(fwd_model, p[:, row])
        key= 'elems_data'
        for col in range(n_col):
            print(row, col)
            im = ax[row, col].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[col][key]),shading='flat', vmin=None,vmax=None)
            title= key + f'#{row}'

            # if np.all(perm <= 1):
            #     title= title +'\nNormalized conductivity distribution'
            # else:
            #     title= title +'\nConductivity distribution'
            ax[row, col].set_title(title)
            ax[row, col].set_xlabel("X axis")
            ax[row, col].set_ylabel("Y axis")
                
            ax[row, col].axis("equal")
            fig.colorbar(im,ax=ax[row, col])

    plt.show(block=False)

def plot_eval_results(results, axis='linear'):

    n_set= len(results)
    n_indic= len(results[0].indicators.keys())

    fig1, ax = plt.subplots(2,n_indic)

    for indx, indic in enumerate(results[0].indicators.keys()):

        ax[0,indx].set_title(indic)
        tmp= list()
        labels= list()
        for res in results:
            tmp.append(np.reshape(res.indicators[indic], (len(res.indicators[indic]),)))
            labels.append(res.info)
        
        
        
        ax[0,indx].boxplot(tmp, labels=labels)
        ax[1,indx].plot(np.array(tmp).T, label=labels)
        ax[1,indx].legend()
    
        
    # plt.plot(mse_nn)
    # plt.plot(rie_nn)
    # plt.plot(icc_nn)
    # plt.plot(mse_eidors)
    # plt.plot(rie_eidors)
    # plt.plot(icc_eidors)
    # plt.show()



    # fig1, ax = plt.subplots(1,3)
    # ax[0].set_title('MSE')
    # ax[0].boxplot((mse_nn, mse_eidors))
    # ax[1].set_title('RIE')
    # ax[1].boxplot((rie_nn, rie_eidors))
    # ax[2].set_title('icc')
    # ax[2].boxplot((icc_nn, icc_eidors))
    plt.show(block=False)
    
if __name__ == "__main__":

    fmdl= loadmat('datasets/test_plot.mat')
    # plot_EIT_mesh(fmdl, fmdl['un2'])

    plot_real_NN_EIDORS(fmdl, fmdl['un2'], fmdl['elem_data'], fmdl['elem_data'])
    # fig1, ax = plt.subplots(2,3)
    # ax[0,0].set_title('MSE')
    # results= [fmdl['un2'],fmdl['un2']]
    # tmp= list()
    # labels= list()
    # for res in results:
    #         # tmp.append(res)
    #         tmp.append(np.reshape(res, (len(res),))) # only vectors
        
    # ax[0,0].boxplot(tmp, labels=['eidors', 'nn'],)
    # ax[1,0].plot(np.array(tmp).T, label=['eidors', 'nn'],)
    # ax[1,0].legend()
    # plt.plot()




    plt.show()

    
    

