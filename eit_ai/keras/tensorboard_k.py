
import logging
import os
from tensorboard import program
from tensorflow.keras.callbacks import TensorBoard

from eit_ai.train_utils.metadata import MetaData
from glob_utils.log.msg_trans  import highlight_msg

logger = logging.getLogger(__name__)

TENSORBOARD_LOG_FOLDER ='log'

def mk_callback_tensorboard(metadata:MetaData):

    log_path= os.path.join(metadata.dir_path,TENSORBOARD_LOG_FOLDER)
    tensorboard = TensorBoard(log_dir= log_path)
    log_tensorboard(log_path)
    return tensorboard

def log_tensorboard(log_path:str):

    tracking_address = log_path # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    logger.info(highlight_msg(f'Tensorflow listening on {url}'))




if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)

