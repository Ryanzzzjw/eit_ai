import numpy as np
from sklearn.metrics import mean_squared_error
from logging import getLogger
from dataclasses import dataclass
logger = getLogger(__name__)


@dataclass
class ImageEIT():
    data:np.ndarray=np.array([])
    label:str=''
    fwd_model:dict=None
@dataclass
class ImageDataset():
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

def EIT_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, multioutput='raw_values').T

def EIT_rie(y_true, y_pred):
    return np.linalg.norm(y_true-y_pred,axis=0)/np.linalg.norm(y_true,axis=0)

# def metrix_rie(y_true, y_pred):    
#     return K.sqrt(K.sum(K.pow(K.abs(y_true-y_pred),2))/K.sum(K.pow(K.abs(y_pred),2)))

# def metrix_re(y_true, y_pred):
#     """Relative error

#     Args:
#         y_true ([type]): [description]
#         y_pred ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
    
#     return K.sqrt(K.sum(K.square(y_true-y_pred))/K.sum(K.square(y_pred)))
    

def EIT_icc(y_true, y_pred):
    icc_matrix = np.corrcoef(y_true, y_pred,rowvar=False) #(Image) Correlation Coefficient (ICC)
    logger.debug(f'ICC_matrix = {icc_matrix}, {icc_matrix.shape}')
    nb_samples=y_true.shape[1]
    I=np.eye(nb_samples)
    return np.diag(icc_matrix[:nb_samples,nb_samples:]*I)

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
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging

    main_log()
    change_level(logging.DEBUG)

    a= None
    print(1 if isinstance(a, int) and a>12 else None)
    # a= np.array([[1 ,2 ,3 ,5, 5, 3], [1 ,2 ,3 ,4, 4, 4],[1 ,2 ,3 ,6, 6, 3]])
    # b= np.array([[1 ,2 ,3 ,3, 3, 3], [1 ,2 ,3 ,3, 4, 3],[1 ,2 ,3 ,3, 6, 3]])
    # print(a.shape, b.shape)

    # error_eval(a,b,True, axis_samples=0)


    # a= np.array([1 ,2 ,3 ,5, 5, 3])
    # b= np.array([1 ,2 ,3 ,3, 3, 3])
    # print(a.shape, b.shape)

    # error_eval(a,b,True, axis_samples=1)

    # a= np.array([1 ,2 ,3 ,4, 4, 4])
    # b= np.array([1 ,2 ,3 ,3, 4, 3])
    # print(a.shape, b.shape)

    # error_eval(a,b,True)

    # a= np.array([1 ,2 ,3 ,6, 6, 3])
    # b= np.array([1 ,2 ,3 ,3, 6, 3])
    # print(a.shape, b.shape)

    # error_eval(a,b,True)








    