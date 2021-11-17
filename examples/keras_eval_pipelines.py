
import os
import matplotlib.pyplot as plt

from eit_tf_workspace.draw_data import *
from eit_tf_workspace.eval_utils import ImageDataset, compute_eval, trunc_img_data_nb_samples
from eit_tf_workspace.raw_data.load_eidors import load_eidors_solution
from eit_tf_workspace.raw_data.raw_samples import reload_samples
from eit_tf_workspace.raw_data.matlab import MatlabSamples
from eit_tf_workspace.train_utils.select_gen import select_gen


from logging import getLogger

from eit_tf_workspace.utils.path_utils import get_POSIX_path
logger = getLogger(__name__)

def std_eval_pipeline(dir_path:str=''):
    logger.info('### Start standard evaluation ###')

    metadata, raw_samples= reload_samples(MatlabSamples(), dir_path=dir_path)
    gen= select_gen(metadata)
    gen.load_model(metadata)
    gen.build_dataset(raw_samples, metadata)

    img_data=[]
    fwd_model=gen.getattr_dataset('fwd_model')
    _, true_img_data=gen.extract_samples(dataset_part='test', idx_samples='all')
    img_data.append(ImageDataset(true_img_data, 'True image',fwd_model))
    logger.info(f'Real perm shape: {true_img_data.shape}')

    nn_img_data = gen.get_prediction(metadata)
    logger.info(f'Predicted perm shape: {nn_img_data.shape}')
    img_data.append(ImageDataset(nn_img_data, 'NN Predicted image',fwd_model))

    # eidors_img_data=load_eidors_solution(
    #     metadata=metadata,
    #     initialdir= os.path.split(metadata.raw_src_file[0])[0])
    
    # for p in eidors_img_data:
    #     img_data.append(ImageDataset(p[0], p[1],fwd_model))

    img_data = trunc_img_data_nb_samples(img_data, max_nb=100) 
    results = compute_eval(img_data) 
    
    plot_compare_samples(image_data=img_data, nb_samples=5, rand=True, orient=Orientation.Portrait)
    plot_compare_samples(image_data=img_data, nb_samples=5, rand=True, orient=Orientation.Landscape)
    # plot_real_NN_EIDORS(gen.getattr_dataset('fwd_model'), true_img_data[randnlist,:].T, nn_img_data[randnlist,:].T)
    plot_eval_results(results, axis='linear')

if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)


    dir_path= 'E:\EIT_Project\05_Engineering\04_Software\Python\eit_tf_workspace\outputs\Std_keras_test_20211117_165710'
    std_eval_pipeline(get_POSIX_path(dir_path))
    plt.show()
    
