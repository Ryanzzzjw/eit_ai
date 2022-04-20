from configparser import Interpolation
import struct
import matplotlib.pyplot as plt
from logging import getLogger
import pyeit.mesh.utils
import pyeit.eit.interp2d
import numpy as np
from eit_ai.draw_data import *
from eit_ai.pytorch.workspace import PyTorchWorkspace
from eit_ai.train_utils.lists import ListPytorchDatasetHandlers, ListPytorchModelHandlers, ListPytorchModels
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import load_samples
from eit_ai.train_utils.metadata import MetaData
import pyeit.eit.greit
import glob_utils.files.matlabfile
import glob_utils.files.files 

logger = getLogger(__name__)
    

def prepare_grid(sz, fwd_model):
    boudary = fwd_model["boundary"]
    nodes = fwd_model["nodes"]
    bnd = np.unique(boudary)
    
    bb_min = np.min(nodes[bnd, :])
    bb_max = np.max(nodes[bnd, :])
    x, y = np.meshgrid(np.linspace(bb_min,bb_max,sz[0]),np.linspace(bb_min,bb_max,sz[1]))
    logger.debug(f'{x=}, {y=}')
    return x, y, bb_min, bb_max  

def plot_samples(fwd_model, perm, U):

    perm=format_inputs(fwd_model, perm)
    U=format_inputs(fwd_model, U)

    tri, pts, data= get_elem_nodal_data(fwd_model, perm)
    
    logger.debug(f'{tri.shape=}, {pts.shape=}, {data=}')
    
    xg, yg, mask = pyeit.eit.interp2d.meshgrid(pts)
    logger.debug(f'{xg.shape=}, {yg.shape=}, {mask.shape=}')
    


if __name__ == "__main__":
    from glob_utils.log.log import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)


    # file_path = r'C:\Users\ryanzzzjw\Desktop\eit_ai\datasets\20220329_001820_2D_cell1_layer1\2D_cell1_layer1_infos2py.mat'
    # var_dict = glob_utils.files.files.load_mat(file_path=file_path)
    # m = glob_utils.files.matlabfile.MatFileStruct()
    # struct = m._extract_matfile(var_dict)

    # fwd_model = struct["fwd_model"]
    
    # sz = np.array([32, 32])
    # prepare_grid(sz, fwd_model)
    # metadata=MetaData()
    # ws = PyTorchWorkspace()# Create a model generator
    # ws.select_model_dataset(
    #     model_handler=ListPytorchModelHandlers.PytorchModelHandler,
    #     dataset_handler=ListPytorchDatasetHandlers.StdPytorchDatasetHandler,
    #     model=ListPytorchModels.StdPytorchModel,
    #     metadata=metadata)

    # metadata.set_ouput_dir(training_name='Std_PyTorch_test', append_date_time= True)
    # metadata.set_4_raw_samples(data_sel= ['Xih','Yih'])
    # metadata._nb_samples = 10
    # raw_samples=load_samples(MatlabSamples(), path, metadata)
    # metadata.set_4_dataset(batch_size=10)
    # ws.build_dataset(raw_samples, metadata)

    # samples_x, samples_y = ws.extract_samples(dataset_part='train', idx_samples=None)
    # plot_samples(ws.getattr_dataset('fwd_model'), samples_y, samples_x)
