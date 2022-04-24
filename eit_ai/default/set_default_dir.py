

from enum import Enum
from glob_utils.directory.inout_dir import DefaultDir, set_default_dir
import os
import pathlib
import logging
logger = logging.getLogger(__name__)

################################################################################
# management of global default directory
################################################################################

DEFAULT_AI_DIR_FILE='ai_default_dirs.txt'
AI_DIRS= DefaultDir()

class AiDirs(Enum):
    matlab_datasets='Matlab datasets'
    ai_models='AI Models'

def set_ai_default_dir(reset:bool= False):
    local_dir= pathlib.Path(__file__).parent.resolve()
    path= os.path.join(local_dir, DEFAULT_AI_DIR_FILE)
    init_dirs={ d.value:'' for d in AiDirs}
    set_default_dir(reset, AI_DIRS, init_dirs, path)

if __name__ == "__main__":
    """"""#
    from glob_utils.log.log import main_log
    main_log()
    set_ai_default_dir(reset=True)  
    print(AiDirs.matlab_datasets.value,AiDirs.ai_models.value)
    print(AI_DIRS.get())
    print(AI_DIRS.get(AiDirs.matlab_datasets.value), AI_DIRS.get(AiDirs.matlab_datasets.value))
    