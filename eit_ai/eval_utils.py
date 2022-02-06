from random import random
import numpy as np
from sklearn.metrics import mean_squared_error
from logging import getLogger
from dataclasses import dataclass
import sys
from eit_ai.default.set_default_dir import AI_DIRS, AiDirs, set_ai_default_dir
from glob_utils.files.files import (FileExt, dialog_get_file_with_ext, find_file, is_file, read_txt,
                                    save_as_mat, save_as_pickle, save_as_txt, save_as_csv, load_csv)
from glob_utils.pth.path_utils import (OpenDialogDirCancelledException,
                                       get_datetime_s, get_dir, get_POSIX_path,
                                       mk_new_dir)
import os
logger = getLogger(__name__)

# Matlab_FILENAME= f'metrics{FileExt.mat}'
Pickle_FILENAME= f'metrics{FileExt.pkl}'
# Txt_FILENAME= f'metrics{FileExt.txt}'
Csv_FILENAME=f'metrics{FileExt.csv}'


@dataclass
class ImageEIT(object):
    data:np.ndarray=np.array([])
    label:str=''
    fwd_model:dict=None
@dataclass
class ImageDataset(object):
    data:np.ndarray=np.array([])
    label:str=''
    fwd_model:dict=None
    def get_single(self, idx)-> ImageEIT:
        return ImageEIT(self.data[idx,:], f'{self.label} #{idx}', self.fwd_model)

class EvalResults(object):
    indicators:dict={}
    info:str=None
    def __init__(self,mse, rie, icc, info) -> None:
        super().__init__()
        self.set_values(mse, rie, icc, info)

    def set_values(self, mse, rie, icc, info='values set')-> None:
        self.indicators['mse']=mse
        self.indicators['rie']=rie
        self.indicators['icc']=icc
        self.info = info

    def save(self, file_path: str=None, export_csv: bool=True):
            """save itself as a pickle and make optional export as txt"""
            # indicators = {k:v.tolist() for k,v in self.indicators.items()}
            # if not self.dir_path:
            #     return
            file_path=file_path
            time = get_datetime_s()
            # matlab_name=os.path.join(dir_path,f'{time}_{Matlab_FILENAME}')
            pickle_name=os.path.join(file_path,f'{time}_{Pickle_FILENAME}')
            # txt_name=os.path.join(dir_path,f'{time}_{Txt_FILENAME}')
            
            
            # save_as_mat(matlab_name, indicators)
            save_as_pickle(pickle_name, self.indicators)
            # save_as_txt(txt_name,indicators)
            
            if export_csv:
                self.export_as_csv(file_path = file_path )
            
    def load_csv(self, file_path: str):
        """load itself and set indicator"""
        if not os.path.isdir(file_path):
            title= 'Select directory of model to evaluate'
            try: 
                file_path=dialog_get_file_with_ext(
                    title=f'Please select *csv files',
                    file_types=[(f"*metrics.csv-files",f"*metrics.csv")]
                )
            except OpenDialogDirCancelledException as e:
                logger.critical('User cancelled the loading')
        
        var = load_csv(file_path=file_path)
        return var
        

    def export_as_csv(self, file_path: str):
        """export data as csv"""
        file_path=file_path
        time = get_datetime_s()
        csv_name=os.path.join(file_path,f'{time}_{Csv_FILENAME}')
        save_as_csv(csv_name, self.indicators)
            
    
def EIT_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, multioutput='raw_values').T

def EIT_rie(y_true, y_pred):
    return np.linalg.norm(y_true-y_pred,axis=0)/np.linalg.norm(y_true,axis=0)
    
def EIT_icc(y_true, y_pred):
    # y_pred=np.random.rand(100,3054).T
    # logger.debug(f'ICC_matrix = {y_true.shape} {y_pred.shape}')
    icc_matrix = np.corrcoef(y_true, y_pred,rowvar=False) #(Image) Correlation Coefficient (ICC)
    logger.debug(f'ICC_matrix = {icc_matrix}, {icc_matrix.shape}')
    nb_samples=y_true.shape[1]
    # I=np.eye(nb_samples)
    return np.diag(icc_matrix[:nb_samples,nb_samples:])

def normalized(y_true, y_pred):

    #!!!I don't know why, but only in that form it normalized correctly, and if I don't put values in [], drops an error: Found input variables with inconsistent numbers of samples: [3054, 1]
    #y_true = tf.keras.utils.normalize(y_true, axis=1) #if I normalize like this, then it is normalized too much (like to 0.02.., not 1), maybe because of [3054, 1]
    y_true = [(y_true-min(y_true))/(max(y_true)-min(y_true))] #normalize 

    print('Real normalized values: ' + str(y_true) + '; Solved normalized values: ' + str(y_pred))

def format_inputs_for_error_eval(y_true:np.ndarray, y_pred:np.ndarray, axis_samples:int):
    logger.debug(f'shape of y_true, y_pred :{y_true.shape}, {y_pred.shape}')  
    if y_true.shape!=y_pred.shape:
        return None, None
    if y_true.ndim==1:
        y_true= np.reshape(y_true, (y_true.shape[0],1))
        y_pred= np.reshape(y_pred, (y_pred.shape[0],1))
    elif axis_samples==0:
        y_true= y_true.T
        y_pred= y_pred.T
    return y_true, y_pred


def error_eval(y_true:np.ndarray, y_pred:np.ndarray, axis_samples:int=0, info:str='set from error_eval'):
    """[summary]

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        axis_samples (int, optional): [description]. Defaults to 0.
        info (str, optional): [description]. Defaults to 'set from error_eval'.

    Returns:
        [type]: [description]
    """  
    y_true, y_pred = format_inputs_for_error_eval(y_true, y_pred, axis_samples)

    mse = EIT_mse(y_true, y_pred) #Mean Squared Error (MSE)
    rie = EIT_rie(y_true, y_pred) #Relative (Image) Error (RIE)
    icc = EIT_icc(y_true, y_pred) #(Image) Correlation Coefficient (ICC)

    logger.info(f'MSE = {mse}, {mse.shape}')
    logger.info(f'RIE = {rie}, {rie.shape}')
    logger.info(f'ICC = {icc}, {icc.shape}')

    return EvalResults(mse, rie, icc, info)

def compute_eval(image_data:list[ImageDataset])-> list[EvalResults]:
    """ compute """   
    # for p in solution:
    #     logger.info('computing max')
    #     if p[0].shape[0]>=max_samples:
    #         max_samples= p[0].shape[0]
    if len(image_data)<2:
        return None
    true= image_data[0]
    results= []
    for pred in image_data[1:]:
        logger.debug(f'Computing evalutaion results: {true.label} VS {pred.label}')
        res=error_eval(true.data, pred.data,info=pred.label)
        results.append(res)
    return results

def get_xshape(n):
    return n.shape[0]

def trunc_img_data_nb_samples(image_data:list[ImageDataset], max_nb:int=None)-> list[ImageDataset]:
    lens=list(map(get_xshape, [row.data for row in image_data]))
    max_samples=max(lens)
    max_samples= max_nb if isinstance(max_nb, int) and max_nb<max_samples else max_samples
    
    trunced_img_data= [ImageDataset(row.data[:max_samples,:], row.label, row.fwd_model) for row in image_data]
    logger.debug(f'nb image trunc to : {max_samples}')
    return trunced_img_data


if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging

    main_log()
    change_level_logging(logging.DEBUG)

    a= None
    print(1 if isinstance(a, int) and a>12 else None)
    # a= np.array([[1 ,2 ,3 ,5, 5, 3], [1 ,2 ,3 ,4, 4, 4],[1 ,2 ,3 ,6, 6, 3]])
    # b= np.array([[1 ,2 ,3 ,3, 3, 3], [1 ,2 ,3 ,3, 4, 3],[1 ,2 ,3 ,3, 6, 3]])
    # print(a.shape, b.shape)

    # error_eval(a,b,True, axis_samples=0)


    a = np.array([1,3,5,6,7])
    b = np.array([2,5,9,6,4])
    Evalres = error_eval(a, b)
    # Evalres.save(dir_path='C:/Users/ryanzzzjw/Desktop/eit_ai/metrics_result')         
    var = Evalres.load_csv('')
    print(var.items())








    