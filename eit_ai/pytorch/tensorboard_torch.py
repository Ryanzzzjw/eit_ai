
import logging
import os
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from eit_ai.train_utils.metadata import MetaData
from glob_utils.log.msg_trans  import highlight_msg

logger = logging.getLogger(__name__)

TENSORBOARD_LOG_FOLDER = 'runs'

def run_tensorboard(logdir_absolute):

   import os, threading
   tb_thread = threading.Thread(
          target=lambda: os.system('/users/ryanzzzjw/anaconda3/envs/'
                                   'torch/bin/tensorboard'
                                   '--logdir=' + TENSORBOARD_LOG_FOLDER),
          daemon=True)
   tb_thread.start()


if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)

    