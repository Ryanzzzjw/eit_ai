from math import nan
from matplotlib import interactive
from pyvista.plotting import scalar_bars
from pyvista.plotting.plotting import Plotter
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import load_samples
from eit_ai.train_utils.metadata import MetaData
import matplotlib.pyplot as plt
from logging import getLogger
import pyeit.mesh.utils
import pyeit.eit.interp2d
import pyvista as pv
import numpy as np
from eit_ai.pytorch.workspace import PyTorchWorkspace
from eit_ai.train_utils.lists import ListPytorchDatasetHandlers, ListPytorchModelHandlers, ListPytorchModels
import vtk
from eit_ai.draw_data import *



logger = getLogger(__name__)

def get_elem_nodal_data(fwd_model, perm, compute:bool=False):
    """ check mesh (tri, pts) in fwd_model and provide elems_data and nodes_data """

    tri = np.array(fwd_model['elems'])
    pts = np.array(fwd_model['nodes'])
    
    # perm= fwd_model['un2']    
    perm= np.reshape(perm, (perm.shape[0],))

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
    if data.ndim == 2:
        tri = np.array(fwd_model['elems'])
        pts = np.array(fwd_model['nodes'])
        if data.shape[1] in [pts.shape[0], tri.shape[0]]:
            data= data.T
    return data
    


def plot_3d(fwd_model, sim, perm):
    
    perm=format_inputs(fwd_model, perm)
    # U=format_inputs(fwd_model, U)
    
    tri, pts, data= get_elem_nodal_data(fwd_model, perm)

    if perm.shape[0]==pts.shape[0]:
        key= 'nodes_data'
    else:
        key= 'elems_data'
        
    sim_data = sim["img_ih"]["elem_data"]
    
    # Faces must contain padding indicating the number of points in the face
    padding = np.ones((tri.shape[0], 1))*tri.shape[1]
    _cells = np.hstack((padding, tri))
    cells = _cells.astype(np.int64).flatten()
    cell_type = np.array([vtk.VTK_TETRA]*tri.shape[0], np.int8)
    chamber = pv.UnstructuredGrid(cells, cell_type, pts)
    
    idx = np.arange(sim_data.shape[0])
    chamber_idx = np.where(sim_data == sim_data.max())
    cell_indices = np.delete(idx, chamber_idx)
    obj = chamber.extract_cells(cell_indices)
    
    pl = pv.Plotter()
    
    pl.add_mesh(chamber, style='wireframe')
    pl.add_mesh(obj, color='blue')
    

def plot_3d_compare_samples(
        image_data:list[ImageDataset],
        nb_samples:int=0,
        rand:bool=False,
        )-> None:

    if not len(image_data):
        logger.warning(f'no ImageData {image_data}')
        return
    
    idx_list= generate_nb_samples2plot(image_data, nb_samples, rand)
    logger.debug(f'{idx_list=}, {idx_list.__len__()=}')
    img2plot= [ImageDataset(id.data[idx_list,:], id.label, id.fwd_model, id.sim) for id in image_data]

    n_img= len(img2plot)
    n_samples= len(idx_list)
    logger.debug(f'{n_img=}, {n_samples=}')

    n_row, n_col=  n_img, n_samples

    pl=pv.Plotter(shape=(n_row, n_col))
    
    for row in range(n_row):
        for col in range(n_col):
            idx_sample, idx_image= col, row
            image=img2plot[idx_image].get_single(idx_sample)
            pl.subplot(row, col)
            plot_3d_EIT_mesh(image, pl)
            
                   
    pl.link_views()
    pl.show()
    
def plot_3d_EIT_mesh(image:ImageEIT, pl:pv.Plotter)-> None:
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
        title= image.label +'\nNorm conduct'
    else:
        title= image.label +'\nConduct'
    
    
    # Faces must contain padding indicating the number of points in the face
    padding = np.ones((tri.shape[0], 1))*tri.shape[1]
    _cells = np.hstack((padding, tri))
    cells = _cells.astype(np.int64).flatten()
    cell_type = np.array([vtk.VTK_TETRA]*tri.shape[0], np.int8)
    chamber = pv.UnstructuredGrid(cells, cell_type, pts)
    
    idx = np.arange(perm.shape[0])
    chamber_idx = np.where(perm == perm.max())
    cell_indices = np.delete(idx, chamber_idx)
    obj = chamber.extract_cells(cell_indices)
    
    
    pl.add_mesh(chamber, style='wireframe')
    pl.add_mesh(obj)
    pl.add_text(title, font_size=20)
    pl.add_axes(interactive=True)
    
    
if __name__ == "__main__":
    from glob_utils.log.log import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)


    