

import os
from logging import getLogger
from eit_ai.constants import EXT_MAT



from eit_ai.raw_data.matlab import load_mat_file
from eit_ai.train_utils.metadata import MetaData
from eit_ai.utils.path_utils import LoadCancelledException
from eit_ai.train_utils.dataset import scale_prepocess

logger = getLogger(__name__)

################################################################################
# Loading of Eidors solution
################################################################################
def load_eidors_solution(metadata:MetaData, initialdir:str='', var_name:str= 'elem_data', ):
    initialdir= initialdir or os.getcwd()

    pred_eidors=[]
    try: 
        for i in range(5):
            title= f'Select {EXT_MAT}-file of eidors solution #{i+1}'
            pred, file_path=load_mat_file(initialdir=initialdir, title=title)
            filename= os.path.splitext(os.path.split(file_path)[1])[0]
            pred_eidors.append([scale_prepocess(pred[var_name].T, metadata.normalize[1]), filename])
    except LoadCancelledException as e :
        logger.info(f'Loading eidors cancelled : ({e})')

    return pred_eidors

                                                                                
if __name__ == "__main__":
    from eit_ai.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    file_path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    # get_matlab_dataset(file_path=file_path, nb_samples2load=10000)
