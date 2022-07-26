### this code is called by EVAL.py

import random
from enum import Enum, auto
import logging
from statistics import median
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.interp2d
import pyeit.mesh.utils
from matplotlib import axes, figure
from scipy.io import loadmat

from eit_ai.eval_utils import EvalResults, ImageDataset, ImageEIT

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True

def get_elem_nodal_data(fwd_model, perm, compute:bool=False):
    """ check mesh (tri, pts) in fwd_model and provide elems_data and nodes_data """

    tri = np.array(fwd_model['elems'])
    pts = np.array(fwd_model['nodes'])
    
    # perm= fwd_model['un2']    
    perm= np.reshape(perm, (perm.shape[0],))

    # tri = tri-1 # matlab count from 1 python from 0
    tri= pyeit.mesh.utils.check_order(pts, tri)

    if perm.shape[0]==pts.shape[0]:
        data={}
        data['elems_data'] = pyeit.eit.interp2d.pts2sim(tri, perm)
        data['nodes_data']= perm
    elif perm.shape[0]==tri.shape[0]:
        data={}
        data['elems_data'] = perm
        data['nodes_data']= pyeit.eit.interp2d.sim2pts(pts, tri, perm)

    for key in data.keys():
        data[key]= np.reshape(data[key], (data[key].shape[0],))
    return tri, pts, data

def format_inputs(fwd_model, data):
    if data.ndim==2:
        tri = np.array(fwd_model['elems'])
        pts = np.array(fwd_model['nodes'])
        if data.shape[1] in [pts.shape[0], tri.shape[0]]:
            data= data.T
    if data.ndim==3:
        data = np.reshape(data,(1, 256))
    return data

def plot_EIT_samples(fwd_model, perm, U):

    perm=format_inputs(fwd_model, perm)
    U=format_inputs(fwd_model, U)

    tri, pts, data= get_elem_nodal_data(fwd_model, perm)

    key = 'nodes_data' if perm.shape[0]==pts.shape[0] else 'elems_data'
    fig, ax = plt.subplots(1,2)
    im = ax[0].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[key]),shading='flat', vmin=None,vmax=None)
    title = key + ('\nNormalized conductivity distribution' if np.all(perm <= 1) else '\nConductivity distribution')

    ax[0].set_title(title)
    ax[0].set_xlabel("X axis")
    ax[0].set_ylabel("Y axis")

    ax[0].axis("equal")
    fig.colorbar(im,ax=ax[0])

    ax[1].plot(U.T)

    plt.show(block=False)

def plot_real_NN_EIDORS(fwd_model, perm_real,*argv):

    _perm = [perm_real]
    for arg in argv:
        if _perm[0].shape==arg.shape:
            _perm.append(arg)

    perm = []
    if perm_real.ndim > 1:
        n_row=  perm_real.shape[1]
        perm.extend(iter(_perm))
    else:
        perm.extend(p.reshape((p.shape[0],1)) for p in _perm)
        n_row= 1
    n_col = len(perm)

    fig, ax = plt.subplots(n_row,n_col)
    if ax.ndim==1:
        ax=ax.reshape((ax.shape[0],1)).T

    key= 'elems_data'
    for row in range(ax.shape[0]):

        data= [dict() for _ in range(n_col)]
        for i, p in enumerate(perm):
            tri, pts, data[i]= get_elem_nodal_data(fwd_model, p[:, row])
        for col in range(n_col):
            print(row, col)
            im = ax[row, col].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[col][key]),shading='flat', vmin=None,vmax=None)
            title = f'{key}#{row}'

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

class Orientation(Enum):
    Portrait=auto()
    Landscape=auto()

def plot_compare_samples(
        image_data:list[ImageDataset],
        nb_samples:int=0,
        rand:bool=False,
        orient:Orientation=Orientation.Portrait)-> None:

    if not len(image_data):
        logger.warning(f'no ImageData {image_data}')
        return
    
    idx_list= generate_nb_samples2plot(image_data, nb_samples, rand)
    logger.debug(f'{idx_list=}, {idx_list.__len__()=}')
    img2plot= [ImageDataset(id.data[idx_list,:], id.label, id.fwd_model, id.sim) for id in image_data]

    n_img= len(img2plot)
    n_samples= len(idx_list)


    n_row, n_col= orient_swap(orient, n_samples, n_img)
    if n_row==1:
        fig, ax = plt.subplots(n_row+1,n_col)
    elif n_col==1:
        fig, ax = plt.subplots(n_row,n_col+1)
    else:
        fig, ax = plt.subplots(n_row,n_col)

    for row in range(n_row):
        for col in range(n_col):
            idx_sample, idx_image= orient_swap(orient, row, col)
            image=img2plot[idx_image].get_single(idx_sample)
            show= [False] * 4
            if idx_sample==0:
                show[0]= True #title
            if col==0 and row==n_row-1:
                show[1]= True #x axis
                show[2]= True #y axis
            fig, ax[row, col], im= plot_EIT_mesh(fig, ax[row, col], image, show)   
            if idx_sample== n_samples-1:
                if orient==Orientation.Landscape:
                    fig.colorbar(im, ax=ax[idx_image, :], location='right', shrink=0.6)
                elif orient==Orientation.Portrait:
                    fig.colorbar(im, ax=ax[:,idx_image], location='bottom', shrink=0.6)
    # fig.set_tight_layout(True)        
    plt.show(block=False)

def orient_swap(orient:Orientation, a:Any, b:Any)-> tuple[Any, Any]:#
    if orient==Orientation.Landscape:
        return b, a
    elif orient==Orientation.Portrait:
        return a, b
    else:
        logger.error(f'wrong orientation type {orient}')
        return a, b

def plot_EIT_mesh(fig:figure.Figure, ax:axes.Axes, image:ImageEIT, show:list[bool]=[True] * 4, colorbar_range:list[int]=[0,1])-> None:
    """[summary]

    Args:
        fig (figure): [description]
        ax (axes): [description]
        image (ImageEIT): [description]
        show (list[bool], optional): [description]. Defaults to [True*4].
    """    
    
    tri, pts, data= get_elem_nodal_data(image.fwd_model, image.data)

    key= 'elems_data'
    perm=np.real(data[key])
    if np.all(perm <= 1) and np.all(perm > 0):
        colorbar_range=[0,1]
        title= image.label +'\nNorm conduct'
    else:
        title= image.label +'\nConduct'
    im = ax.tripcolor(pts[:,0], pts[:,1], tri, perm, shading='flat', vmin=colorbar_range[0],vmax=colorbar_range[1])
    # ax.axis("equal")
    # fig.set_tight_layout(True)
    # ax.margins(x=0.0, y=0.0)
    ax.set_aspect('equal', 'box')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.axis('off')
    if show[0]:
        ax.set_title(title)
    if show[1]:
        ax.axis('on')
        ax.set_xlabel("X axis")
    if show[2]:
        ax.set_ylabel("Y axis")
    if show[3]:    
        fig.colorbar(im,ax=ax)
    return fig, ax, im
    

   
def generate_nb_samples2plot(
        image_data:list[ImageDataset],
        nb_samples:Union[int,list[int]]=3,
        rand:bool=False) -> list[int]:
    """ """
    nb_samples_total=image_data[0].data.shape[0]
    if nb_samples_total==0:
        logger.error('image data do not contain any data!!!!)')
        return None

    if isinstance(nb_samples, list):
        if max(nb_samples)>nb_samples_total:
            logger.error(f'List of indexes : {nb_samples} is not correc')
            logger.info('first image will be plot')
            return [0]
        return nb_samples
    elif isinstance(nb_samples, int):
        if nb_samples==0:
            nb_samples=1
        if nb_samples>nb_samples_total:
            logger.error(f'List of indexes : {nb_samples=}>{nb_samples_total=}')
            nb_samples=1

        if rand:
            return random.sample(range(nb_samples_total), nb_samples)
        else:
            nb_samples_total=nb_samples
        return range(nb_samples_total)


def plot_eval_results(results:list[EvalResults], axis='linear', plot_type=None):

    n_set= len(results)
    n_indic= len(results[0].indicators.keys())

    fig, ax = plt.subplots(1,n_indic)

    for indx, indic in enumerate(results[0].indicators.keys()):

        ax[indx].set_title(indic)
        tmp= []
        labels=[]
        for res in results:
            tmp.append(np.reshape(res.indicators[indic], (len(res.indicators[indic]),)))
            labels.append(res.info)
        bp = ax[indx].boxplot(tmp, labels=labels, showmeans=True)
        
        # means = [round(item.get_ydata()[0], 3) for item in bp['means']]
        # medians = [round(item.get_ydata()[0], 3) for item in bp['medians']]
        
        # for i, line in enumerate(bp['medians']):
        #     x, y = line.get_xydata()[1]
        #     text = ' mean={:.3f}\n med={:.3f}'.format(means[0], medians[0])
        #     ax[indx].annotate(text, xy=(x, y))
        # ax[1,indx].plot(np.array(tmp).T, label=labels)
        # ax[1,indx].legend()
    
    plt.show(block=False)
    
if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)

    print()
    print([True for _ in range(4)])

    
    

