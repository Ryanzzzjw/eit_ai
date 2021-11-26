

################################################################################
# 
################################################################################
FORMAT_DATE_TIME= "%Y%m%d_%H%M%S"
FORMAT_TIME= "%Hh %Mm %Ss"
DEFAULT_OUTPUTS_DIR= 'outputs'



################################################################################
# 
################################################################################
EXT_MAT= '.mat'
EXT_PKL= '.pkl'
EXT_TXT= '.txt'


EXT_IDX_FILE= 'idx_samples'
# EXT_EIDORS_SOLVING_FILE= '_test_elem_data'+ EXT_MAT

################################################################################
# 
################################################################################

METADATA_FILENAME='metadata' + EXT_TXT
MODEL_SUMMARY_FILENAME='model_summary' + EXT_TXT
TENSORBOARD_LOG_FOLDER ='log'


if __name__ == "__main__":
    from eit_ai.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)