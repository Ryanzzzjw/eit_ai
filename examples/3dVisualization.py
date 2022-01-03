from numpy.core.fromnumeric import ndim
from numpy.core.shape_base import block
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
        if data.shape[1]==pts.shape[0] or data.shape[1]==tri.shape[0]:
            data= data.T
    return data
    


def plot_3d(fwd_model, perm, U):
    
    perm=format_inputs(fwd_model, perm)
    U=format_inputs(fwd_model, U)
    
    tri, pts, data= get_elem_nodal_data(fwd_model, perm)

    if perm.shape[0]==pts.shape[0]:
        key= 'nodes_data'
    else:
        key= 'elems_data'
    
    # fig, ax = plt.subplots(1,2)
    im = pv.PolyData(pts)
    im.plot(cpos='xy', show_edges=True)


if __name__ == "__main__":
    from glob_utils.log.log import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)

    debug = True

    if debug:
        path = r'C:\Users\ryanzzzjw\Downloads/eit_ai/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset' \
               '/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat '
    else:
        path = ''
    
    metadata=MetaData()
    ws = PyTorchWorkspace()# Create a model generator
    ws.select_model_dataset(
        model_handler=ListPytorchModelHandlers.PytorchModelHandler,
        dataset_handler=ListPytorchDatasetHandlers.StdPytorchDatasetHandler,
        model=ListPytorchModels.StdPytorchModel,
        metadata=metadata)

    metadata.set_ouput_dir(training_name='Std_PyTorch_test', append_date_time= True)
    metadata.set_4_raw_samples(data_sel= ['Xih','Yih'])
    # metadata._nb_samples = 10000
    raw_samples=load_samples(MatlabSamples(), path, metadata)
    metadata.set_4_dataset(batch_size=1000)
    ws.build_dataset(raw_samples, metadata)

    samples_x, samples_y = ws.extract_samples(dataset_part='train', idx_samples=None)
    plot_3d(ws.getattr_dataset('fwd_model'), samples_y, samples_x)
    

    

