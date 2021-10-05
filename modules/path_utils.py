import os

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory


def mk_ouput_dir(name, verbose= True, default_out_dir= 'outputs'):
    if not os.path.isdir(default_out_dir):
        os.mkdir(default_out_dir)

    output_dir= os.path.join(default_out_dir, name)

    if verbose:
        print('\nResults are to found in:\n >> {}'.format(output_dir))

    os.mkdir(output_dir)

    return output_dir

def get_dir(initialdir=os.getcwd(), title='Select a directory'):

    Tk().withdraw()
    path_dir = askdirectory(initialdir=initialdir, title= title) 
    return path_dir


def verify_file(path, extension):
    path_out=""
    if os.path.isfile(path):
            _, file_extension = os.path.splitext(path)
            if file_extension==extension:
                path_out= path
    return path_out