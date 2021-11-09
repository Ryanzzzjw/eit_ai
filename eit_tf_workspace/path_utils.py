import os

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
import pickle
import json
import datetime
from typing import List, Union

from matplotlib.pyplot import title
import  eit_tf_workspace.constants as const

def get_date_time():
    _now = datetime.datetime.now()
    return _now.strftime(const.FORMAT_DATE_TIME)

def get_POSIX_path(path:str):

    return path.replace('\\','/')


def mk_ouput_dir(name, verbose= True, default_out_dir= const.DEFAULT_OUTPUTS_DIR ):
    """[summary]

    Args:
        name ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.
        default_out_dir (str, optional): [description]. Defaults to 'outputs'.

    Returns:
        [type]: [description]
    """
    if not os.path.isdir(default_out_dir):
        os.mkdir(default_out_dir)

    output_dir= os.path.join(default_out_dir, name)

    if verbose:
        print('\nResults are to found in:\n >> {}'.format(output_dir))

    os.mkdir(output_dir)

    return output_dir

def get_dir(initialdir=None, title='Select a directory'):
    """[summary]

    Args:
        initialdir ([type], optional): [description]. Defaults to os.getcwd().
        title (str, optional): [description]. Defaults to 'Select a directory'.

    Returns:
        [type]: [description]
    """
    Tk().withdraw()
    initialdir = initialdir or os.getcwd()
    return askdirectory(initialdir=initialdir, title= title)

def get_file(filetypes:List[tuple[str, str]]=[("All files","*.*")], verbose:bool= True, initialdir:str=None, title:str= ''):
    """used to get select files using gui (multiple types of file can be set!)

    Args:
        filetypes (list, optional): [description]. Defaults to [("All files","*.*")].
        verbose (bool, optional): [description]. Defaults to True.
        path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

    initialdir = initialdir or os.getcwd()

    whole_path = askopenfilename(
                    initialdir=initialdir,
                    filetypes=filetypes,
                    title=title) # show an "Open" dialog box and return the path to the selected file
    path, filename = os.path.split(whole_path)
    if verbose:
        print(path, filename)
    return path, filename

def verify_file(path, extension, debug=False):
    """[summary]

    Args:
        path ([type]): [description]
        extension ([type]): [description]

    Returns:
        [type]: [description]
    """
    path_out=""
    if debug:
        print(os.path.isfile(path))
    if os.path.isfile(path):
            _, file_extension = os.path.splitext(path)
            if debug:
                print(os.path.splitext(path),file_extension)
            if file_extension==extension:
                path_out= path
    return path_out

def save_as_pickle(filename, class2save, verbose=True, add_ext=True):
    """[summary]

    Args:
        filename ([type]): [description]
        class2save ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.
    """
    if add_ext:
        filename= os.path.splitext(filename)[0] + const.EXT_PKL

    with open(filename, 'wb') as file:
        pickle.dump(class2save, file, pickle.HIGHEST_PROTOCOL)
    print_saving_verbose(filename, class2save, verbose)

def save_as_txt(filepath, class2save, verbose=True, add_ext=True):
    """[summary]

    Args:
        filename ([type]): [description]
        class2save ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.
        add_ext (bool, optional): [description]. Defaults to True.
    """
    if add_ext:
        filepath= os.path.splitext(filepath)[0] + const.EXT_TXT
    list_of_strings = []
    if isinstance(class2save,str):
        list_of_strings.append(class2save)
    elif isinstance(class2save, list):
        for item in class2save:
            list_of_strings.append(f'{item}')
    else:

        tmp_dict= class2save.__dict__
        list_of_strings.append('Dictionary form:')
        list_of_strings.append(json.dumps(class2save.__dict__))
        list_of_strings.append('\n\nSingle attributes:')
        list_of_strings.extend([f'{key} = {tmp_dict[key]},' for key in class2save.__dict__ ])

    with open(filepath, 'w') as file:
        [ file.write(f'{st}\n') for st in list_of_strings ]

    print_saving_verbose(filepath, class2save, verbose)

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

def load_pickle(filename, class2upload=None, verbose=True):
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
    print_loading_verbose(filename, loaded_class, verbose)
    if not class2upload:
        return loaded_class

    for key in loaded_class.__dict__.keys():
            setattr(class2upload, key, getattr(loaded_class,key))
    return class2upload

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


if __name__ == "__main__":
   
    path_pkl='E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset/2D_16e_adad_cell3_SNR20dB_50k_infos2py.pkl'
    # path_pkl=path_pkl.replace('/','\\')
    print(verify_file(path_pkl, extension=const.EXT_PKL, debug=True))

    a= 'print_saving_verbose'
    print(os.path.splitext('hhhhhhhh'))
    if os.path.splitext('hhhhhhhh')[1]:
        print_saving_verbose('ffffffffffffffffffffff', class2save= None, verbose=True)

    path= "E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/DStest/test10_infos2py.mat" 

    print(os.path.split(os.path.split(path)[0]))