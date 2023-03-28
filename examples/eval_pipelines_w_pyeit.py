from numpy import ndarray
from eit_ai.draw_3d import *
from eit_ai.draw_3d import plot_3d_compare_samples
from eit_ai.draw_data import *
import matplotlib.pyplot as plt
from eit_ai.eval_utils import ImageDataset, compute_eval, trunc_img_data_nb_samples

from eit_ai.raw_data.raw_samples import reload_samples
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.train_utils.metadata import reload_metadata
from eit_ai.train_utils.select_workspace import select_workspace
from eit_model.model import EITModel
from eit_model.data import EITFrameMeasuredChannelVoltage
from eit_model.reconstruction import EITReconstruction, EITReconstructionData
from eit_model.solver_pyeit import PyEitRecParams, SolverPyEIT
from eit_model.plot import EITImage2DPlot
from eit_model.imaging import build_EITImaging
import logging

logger = logging.getLogger(__name__)



def set_pyeit(model_path:str= None) -> EITReconstruction:

    params= PyEitRecParams(
        mesh_generation_mode_2D=True,
        normalize=False # true data are already normalized
        
    )
    
    rec= EITReconstruction()
    eit_mdl = EITModel()
    # eit_mdl.load_matfile(model_path)
    eit_mdl.load_defaultmatfile()
    eit_mdl.set_refinement(0.2)
    rec.init_eit_model(eit_mdl)
    rec.init_solver(SolverPyEIT, params)
    rec.imaging= build_EITImaging(eit_imaging= "Time difference imaging", transform="Real", show_abs= False)

    return rec

def rec_image(rec:EITReconstruction, X_data:ndarray, nb:int= 1000 )-> ImageDataset:

    rec.enable_rec()
    data= np.zeros((nb, rec.eit_model.fem.elems.shape[0]))
    for i in range(nb):
        raw_voltage=X_data[i,:]
        logger.info(f"{raw_voltage=}")
        raw_voltage= raw_voltage.reshape((16,16))
        logger.info(f"{raw_voltage=}")
        raw_voltage= np.concatenate((raw_voltage, np.zeros_like(raw_voltage)), axis =1)
        logger.info(f"{raw_voltage=}")
        raw_voltage= raw_voltage.reshape((16,32))
        logger.info(f"{raw_voltage=}")
        ref_frame= EITFrameMeasuredChannelVoltage(np.zeros_like(raw_voltage),f"refFrame {i}")
        meas_frame= EITFrameMeasuredChannelVoltage(raw_voltage,f"measFrame {i}")
        d=EITReconstructionData(ref_frame, meas_frame)
        rec.rec_process(d)
        eit_img, _, _=rec.imaging_results()
        plot(eit_img)
        data[i,:]= eit_img.data.reshape((1,-1))

    

    fwd_model= {
        'elems':eit_img.elems,
        'nodes': eit_img.nodes,
    }
    sim= {}

    return ImageDataset(np.array(data).reshape(nb, -1), "PyEIT", fwd_model, sim)

def plot(img_rec):

    fig, ax = plt.subplots(1,1)
    img_graph= EITImage2DPlot()
    img_graph.plot(fig,ax,img_rec)
    plt.show(block= False)



def eval_pipeline(dir_path:str=''):
    
    rec = set_pyeit()

    logger.info('### Start standard evaluation ###')
    
    metadata = reload_metadata(dir_path=dir_path)
    # metadata.set_4_raw_samples(data_sel= ['Xihn-Xhn','Yih-Yh'])
    raw_samples= reload_samples(MatlabSamples(),metadata)
    ws= select_workspace(metadata)
    ws.load_model(metadata)
    ws.build_dataset(raw_samples, metadata)

    img_data=[]
    fwd_model=ws.getattr_dataset('fwd_model')
    sim=ws.getattr_dataset('sim')
    true_data, true_img_data=ws.extract_samples(dataset_part='test', idx_samples='all')
    img_data.append(ImageDataset(true_img_data, 'True image',fwd_model, sim))
    logger.info(f'Real perm shape: {true_img_data.shape}')
    logger.info(f'Data shape: {true_data.shape}')

    img_dataset= rec_image(rec, true_data, nb= 5)
    # img_data.append(img_dataset)

    
    nn_img_data = ws.get_prediction(metadata)
    # TODO make a reshape of nn_img_data
    nn_img_data=nn_img_data.reshape(true_img_data.shape)
    #plt.plot(nn_img_data.T)
    logger.info(f'Predicted perm shape: {nn_img_data.shape}')
    img_data.append(ImageDataset(nn_img_data, 'NN Predicted image',fwd_model, sim))



    img_data = trunc_img_data_nb_samples(img_data, max_nb=5) 
    # results = compute_eval(img_data)

    # results[0].save(file_path='C:/Users/ryanzzzjw/Desktop/eit_ai/metrics_result')
    # print(results[0].indicators['mse'])
    
    # plot_eval_results(results, axis='linear')
    plot_compare_samples(image_data=img_data, nb_samples=5, orient=Orientation.Portrait)
    # plot_compare_samples(image_data=img_data, nb_samples=5, orient=Orientation.Landscape)
    # plot_3d_compare_samples(image_data=img_data, nb_samples=1)
    # plot_real_NN_EIDORS(gen.getattr_dataset('fwd_model'), true_img_data[randnlist,:].T, nn_img_data[randnlist,:].T)



def test_single_predict(dir_path:str=''):
    logger.info('### Start standard evaluation ###')

    metadata = reload_metadata(dir_path=dir_path)
    raw_samples= reload_samples(MatlabSamples(),metadata)
    ws= select_workspace(metadata)
    ws.load_model(metadata)
    ws.build_dataset(raw_samples, metadata)

    img_data=[]
    fwd_model=ws.getattr_dataset('fwd_model')
    single_X, true_img_data=ws.extract_samples(dataset_part='test', idx_samples='all')
    img_data.append(ImageDataset(true_img_data, 'True image',fwd_model))
    logger.info(f'Real perm shape: {true_img_data.shape}')
    logger.info(f'single_X shape: {single_X.shape=}')
    # nn_img_data = gen.get_prediction(metadata)
    # logger.info(f'Predicted perm shape: {nn_img_data.shape}')
    # img_data.append(ImageDataset(nn_img_data, 'NN Predicted image',fwd_model))
    
    single= single_X[0,:]
    # plt.plot(single)
    logger.info(f'Real perm shape: {single_X.shape}')
    nn_img_data = ws.get_prediction(metadata,single_X=single)
    logger.info(f'Predicted perm shape: {nn_img_data.shape}')
    img_data.append(ImageDataset(nn_img_data, 'NN Predicted image #1',fwd_model))
    single= single_X[1,:]
    # plt.plot(single)
    logger.info(f'Real perm shape: {single_X.shape}')
    nn_img_data = ws.get_prediction(metadata,single_X=single, preprocess=True)
    logger.info(f'Predicted perm shape: {nn_img_data.shape}')
    img_data.append(ImageDataset(nn_img_data, 'NN Predicted image #2',fwd_model))
    single= single_X[2,:]
    # plt.plot(single)
    logger.info(f'Real perm shape: {single.shape}')
    nn_img_data = ws.get_prediction(metadata,single_X=single)
    logger.info(f'Predicted perm shape: {nn_img_data.shape}')
    img_data.append(ImageDataset(nn_img_data, 'NN Predicted image #3',fwd_model))

    img_data = trunc_img_data_nb_samples(img_data, max_nb=1) 
    results = compute_eval(img_data)  
    
    plot_compare_samples(image_data=img_data, orient=Orientation.Portrait)
    plot_compare_samples(image_data=img_data, orient=Orientation.Landscape)
    # plot_real_NN_EIDORS(gen.getattr_dataset('fwd_model'), true_img_data[randnlist,:].T, nn_img_data[randnlist,:].T)
    plot_eval_results(results, axis='linear')

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    from glob_utils.directory.utils import get_POSIX_path    
    import logging
    main_log()
    change_level_logging(logging.INFO)

    eval_pipeline('')
    # dir_path= 'E:\EIT_Project\05_Engineering\04_Software\Python\eit_ai\outputs\Std_keras_test_20211117_165710'
    # test_single_predict('')
    plt.show()   