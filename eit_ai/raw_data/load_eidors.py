

import os
from logging import getLogger

from glob_utils.files.files import FileExt, OpenDialogFileCancelledException
from numpy import ndarray
from eit_ai.raw_data.matlab import load_mat_file
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.dataset import scale_prepocess

logger = getLogger(__name__)

################################################################################
# Loading of Eidors solution
################################################################################
def load_eidors_solution(
    metadata:MetaData, 
    initialdir:str=None, 
    var_name:str= 'elem_data')->list[list[ndarray,str]]:

    """Load EIDORS solutions, which are stored in mat-files under the 
    variable key "elem-data",

    up to 5 solution can be loaded

    Args:
        metadata (MetaData): metadata from AI workspace
        initialdir (str, optional): directory for Open Dilaog Box. 
        Defaults to `None` (will be set to cwd).
        var_name (str, optional): variable key. Defaults to 'elem_data'.

    Returns:
        list[list[ndarray,str]]: List of (loaded eidors solution, file_name),
        loaded eidors solution is an array like (nb_samples, feature)
    """    
    pred_eidors=[]
    try: 
        for i in range(5):
            title= f'Select {FileExt.mat}-file of eidors solution #{i+1}'
            pred, file_path=load_mat_file(initialdir=initialdir, title=title)
            file_name= os.path.splitext(os.path.split(file_path)[1])[0]
            pred_eidors_i= scale_prepocess(
                pred[var_name].T,
                metadata.normalize[1])
            pred_eidors.append([pred_eidors_i, file_name])
    except OpenDialogFileCancelledException as e :
        logger.info(f'Loading eidors cancelled : ({e})')

    return pred_eidors

                                                                                
if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    file_path='E:/EIT_Project/05_Engineering/04_Software/Python/eit_app/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat'
    # get_matlab_dataset(file_path=file_path, nb_samples2load=10000)
