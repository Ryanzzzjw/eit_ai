
from logging import getLogger
import os
from tensorboard import program
from tensorflow.keras.callbacks import TensorBoard

import eit_tf_workspace.constants as const
from eit_tf_workspace.train_utils.metadata import MetaData

logger = getLogger(__name__)


def mk_callback_tensorboard(metadata:MetaData):

    log_path= os.path.join(metadata.ouput_dir,const.TENSORBOARD_LOG_FOLDER)
    tensorboard = TensorBoard(log_dir= log_path)
    log_tensorboard(log_path)

    return tensorboard

def log_tensorboard(log_path:str):

    tracking_address = log_path # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"\n######################################\nTensorflow listening on {url}\n######################################\n")




if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)

