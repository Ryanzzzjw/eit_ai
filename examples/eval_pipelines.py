
import matplotlib.pyplot as plt

from eit_ai.draw_3d import *
from numpy import block, ndarray
from sklearn import metrics
from eit_ai.draw_3d import plot_3d_compare_samples
from eit_ai.draw_data import *
from eit_ai.eval_utils import ImageDataset, compute_eval, trunc_img_data_nb_samples

from eit_ai.raw_data.raw_samples import reload_samples
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.train_utils.metadata import reload_metadata
from eit_ai.train_utils.select_workspace import select_workspace
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)
writer = SummaryWriter()


def eval_pipeline(dir_path:str=''):
    logger.info('### Start standard evaluation ###')
    
    metadata = reload_metadata(dir_path=dir_path)
    raw_samples= reload_samples(MatlabSamples(),metadata)
    ws= select_workspace(metadata)
    ws.load_model(metadata)
    ws.build_dataset(raw_samples, metadata)

    img_data=[]
    fwd_model=ws.getattr_dataset('fwd_model')
    sim=ws.getattr_dataset('sim')
    _, true_img_data=ws.extract_samples(dataset_part='test', idx_samples='all')
    img_data.append(ImageDataset(true_img_data, 'True image',fwd_model, sim))
    logger.info(f'Real perm shape: {true_img_data.shape}')

    nn_img_data = ws.get_prediction(metadata)
    # TODO make a reshape of nn_img_data
    nn_img_data=nn_img_data.reshape(true_img_data.shape)
    #plt.plot(nn_img_data.T)
    logger.info(f'Predicted perm shape: {nn_img_data.shape}')
    img_data.append(ImageDataset(nn_img_data, 'NN Predicted image',fwd_model, sim))

    
    # eidors_img_data=load_eidors_solution(
    #     metadata=metadata,
    #     initialdir= os.path.split(metadata.raw_src_file[0])[0])
    
    # for p in eidors_img_data:
    #     img_data.append(ImageDataset(p[0], p[1],fwd_model))

    img_data = trunc_img_data_nb_samples(img_data, max_nb=100) 
    results = compute_eval(img_data)
    
    # results[0].save(file_path='C:/Users/ryanzzzjw/Desktop/eit_ai/metrics_result')
    # print(results[0].indicators['mse'])
    
    plot_eval_results(results, axis='linear')
    # plot_compare_samples(image_data=img_data, nb_samples=5, orient=Orientation.Portrait)
    # plot_compare_samples(image_data=img_data, nb_samples=5, orient=Orientation.Landscape)
    plot_3d_compare_samples(image_data=img_data, nb_samples=3)
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

    # eidors_img_data=load_eidors_solution(
    #     metadata=metadata,
    #     initialdir= os.path.split(metadata.raw_src_file[0])[0])
    
    # for p in eidors_img_data:
    #     img_data.append(ImageDataset(p[0], p[1],fwd_model))

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
    change_level_logging(logging.DEBUG)

    eval_pipeline('')
    # dir_path= 'E:\EIT_Project\05_Engineering\04_Software\Python\eit_ai\outputs\Std_keras_test_20211117_165710'
    # test_single_predict('')
    plt.show()   