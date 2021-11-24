import os

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
import pickle
import json
import datetime
from typing import Union

import  eit_tf_workspace.constants as const

from logging import getLogger
from eit_tf_workspace.utils.files import FileExt

logger = getLogger(__name__)
class DialogCancelledException(Exception):
    """"""
FORMAT_DATE_TIME= "%Y%m%d_%H%M%S"

def get_date_time()->str:
    _now = datetime.datetime.now()
    return _now.strftime(FORMAT_DATE_TIME)

def get_POSIX_path(path:str)->str:

    return path.replace('\\','/')


def dir_exist(dir_path:str, create_auto:bool=False)->bool:
    """Test if a directory exist
    setting the create argument to `True` allow the automatic creation
    of the directory 

    Args:
        dir_path (str): directory path to test / create
        create_auto (bool, optional): allow the automatic creation
    of the directory if it not exist . Defaults to False.

    Returns:
        bool: return `True` if dir_path is an existing dir or
        if create_auto is set to `True`
    """    

    exist= os.path.isdir(dir_path)
    if not exist and create_auto:
        os.mkdir(dir_path)
        logger.info(f'Directory: {dir_path} - created')
        exist= True
    return exist


def mk_new_dir(dir_name:str, parent_dir:str= None )-> str:
    """Create a directory in a specified parent directory
    parent_dir is not specified the new dir will be created 
    in the current working directory (cwd)
    Args:
        dir_name (str): name of the new directory to create
        parent_dir (str, optional): . Defaults to None.

    Returns:
        [str]: the path of the created directory
    """    
    if not parent_dir:
        parent_dir=os.getcwd()

    dir_exist(parent_dir, create_auto=True)

    new_dir_path= os.path.join(parent_dir, dir_name)
    os.mkdir(new_dir_path)

    return new_dir_path

def get_dir(title:str='Select a directory', initialdir:str=None)->str:
    """Open an explorer dialog for selection of a directory

    Args:
        title (str, optional): title of the Dialog . Defaults to 'Select a directory'.
        initialdir (str, optional): path of initial directory for the explorer dialog. Defaults to None.

    Raises:
        DialogCancelledException: when user cancelled the dialog 

    Returns:
        str: a directory path selected by a user
    """    
    
    Tk().withdraw()
    # show an "Open" dialog box and return the path to the selected directory
    dir_path = askdirectory(
        initialdir=initialdir or os.getcwd(),
        title= title)
    if not dir_path :
        raise DialogCancelledException()
    return dir_path    

def get_file(file_types=[("All files","*.*")], verbose:bool= False, initialdir:str=None, title:str= '', split:bool=True):
    """used to get select files using gui (multiple types of file can be set!)

    Args:
        filetypes (list, optional): [description]. Defaults to [("All files","*.*")].
        verbose (bool, optional): [description]. Defaults to True.
        path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

    # show an "Open" dialog box and return the path to the selected file
    file_path = askopenfilename(
        initialdir=initialdir or os.getcwd(),
        filetypes=file_types,
        title=title) 
    print(file_path)
    if not file_path:
        raise DialogCancelledException()
    if not split:
        return file_path
    path, filename = os.path.split(file_path)

    return path, filename
    
def verify_file(file_path:str, extension:Union[str, FileExt]):
    """[summary]

    Args:
        path ([type]): [description]
        extension ([type]): [description]

    Returns:
        [type]: [description]
    """
    path_out=''
    if not os.path.isfile(file_path):
        logger.debug(f'File "{file_path}" does not exist or is not a file!')
        return path_out
    _, file_extension = os.path.splitext(file_path)
    if file_extension== f'{extension}':
        path_out= file_path
    return path_out

def save_as_pickle(file_path, class2save, verbose=False, append_ext=True):
    """
    """
    file_path= append_extention(file_path, FileExt.pkl) if append_ext else file_path
    with open(file_path, 'wb') as file:
        pickle.dump(class2save, file, pickle.HIGHEST_PROTOCOL)
    print_saving_verbose(file_path, class2save, verbose)

def save_as_txt(file_path, class2save, verbose=True, add_ext=True):
    """[summary]

    Args:
        filename ([type]): [description]
        class2save ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.
        add_ext (bool, optional): [description]. Defaults to True.
    """
    file_path= append_extention(file_path, FileExt.txt) if add_ext else file_path
    
    list_of_strings = []
    if isinstance(class2save,str):
        list_of_strings.append(class2save)
    elif isinstance(class2save, list):
        for item in class2save:
            list_of_strings.append(f'{item}')
    elif isinstance(class2save, dict):
        list_of_strings.append('Dictionary form:')
        list_of_strings.append(json.dumps(class2save))
        list_of_strings.append('\n\nSingle attributes:')
        list_of_strings.extend([f'{key} = {class2save[key]},' for key in class2save ])      
    else:
        tmp_dict= class2save.__dict__
        list_of_strings.append('Dictionary form:')
        list_of_strings.append(json.dumps(class2save.__dict__))
        list_of_strings.append('\n\nSingle attributes:')
        single_attrs= [f'{key} = {tmp_dict[key]}' for key in class2save.__dict__ ]
        single_attrs= [ attr if len(attr)< 200 else f'{attr[:200]}...' for attr in single_attrs]
        list_of_strings.extend(single_attrs)

    with open(file_path, 'w') as file:
        [ file.write(f'{st}\n') for st in list_of_strings ]

    # print_saving_verbose(filepath, class2save, verbose)

def append_extention(file_path:str, ext:Union[str, FileExt])->str:
    """ Append and or replace extention to the file path
    
    Args:
        filepath (str): path of the file
        ext (Union[str, FileExt]): extention 

    Returns:
        str: path of the file with extention
    """    
    return f'{os.path.splitext(file_path)[0]}{ext}'

def read_txt(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    if 'Dictionary form:' in lines[0]:
        return json.loads(lines[1].replace('\n', ''))

def print_saving_verbose(filename, class2save= None, verbose=True):
    """[summary]

    Args:
        filename ([type]): [description]
        class2save ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): [description]. Defaults to True.
    """

    if verbose:
        if hasattr(class2save, 'type'):
            print('\n{} saved in : ...{}'.format(class2save.type, filename[-50:]))
        else:
            print('\n Some data were saved in : ...{}'.format(filename[-50:]))

def load_pickle(filename, class2upload=None):
    """[summary]

    Args:
        filename ([type]): [description]
        class2upload ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    with open(filename, 'rb') as file:
        loaded_class = pickle.load(file)
    # print_loading_verbose(filename, loaded_class, verbose)
    if not class2upload:
        return loaded_class
    set_existing_attrs(class2upload, loaded_class)
    return class2upload

def set_existing_attrs(class2upload, newclass):

    for key in newclass.__dict__.keys():
            if key in class2upload.__dict__.keys():
                setattr(class2upload,key, getattr(newclass,key))

    # for key in class2upload.__dict__.keys():
    #         setattr(class2upload, key, getattr(newclass,key))


def print_loading_verbose(filename, classloaded= None, verbose=True):
    """[summary]

    Args:
        filename ([type]): [description]
        classloaded ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): [description]. Defaults to True.
    """
    if verbose:
        if hasattr(classloaded, 'type'):
            print('\n{} object loaded from : ...{}'.format(classloaded.type, filename[-50:]))
        else:
            print('\nSome data were loaded from : ...{}'.format(filename[-50:]))

class LoadCancelledException(Exception):
    """"""
class WrongFileTypeSelectedError(Exception):
    """"""

def get_file_dir_path( file_path:str='', extension:str=FileExt.txt.value, **kwargs):
    """

    Args:
        file_path (str, optional): [description]. Defaults to ''.
        extension ([type], optional): [description]. Defaults to const.EXT_MAT.

    Raises:
        LoadCancelledException: [description]
        WrongFileTypeSelectedError: [description]

    Returns:
        [type]: [description]
    """
    title= kwargs.pop('title') if 'title' in kwargs else None # pop title

    file_path= verify_file(file_path, extension=extension)        
    if not file_path:
        try: 
            file_path =get_file(
                title=title or f'Please select *{extension} files',
                file_types=[(f"{extension}-file",f"*{extension}")],
                split=False,**kwargs)
        except DialogCancelledException:
            raise LoadCancelledException('Loading aborted from user')
    dir_path=os.path.split(file_path)[0]

    if not verify_file(file_path, extension=extension):
        raise WrongFileTypeSelectedError('User selected wrong file!')

    return dir_path , file_path




if __name__ == "__main__":
    from glob_utils.log.log  import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)

    a= {
        'ab':1,
        'cd':2
    }

    print(a.pop('DF') if 'DF' in a else None)
    print(a)
   
    path_pkl='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    # path_pkl=path_pkl.replace('/','\\')
    print(verify_file(path_pkl, extension=const.EXT_PKL))

    a= 'print_saving_verbose'
    print(os.path.splitext('hhhhhhhh'))
    if os.path.splitext('hhhhhhhh')[1]:
        print_saving_verbose('ffffffffffffffffffffff', class2save= None, verbose=True)

    path= "E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/DStest/test10_infos2py.mat" 

    print(os.path.split(os.path.split(path)[0]))