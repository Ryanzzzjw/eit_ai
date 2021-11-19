

import os
import pickle
from logging import error
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from typing import Any, List

import numpy as np
import sklearn.model_selection
import tensorflow as tf
from scipy.io import loadmat
from scipy.io.matlab.mio import savemat
import modules.interp2d as interp2d

from modules.path_utils import *
# from modules.load_mat_files import *
# from load_mat_files import *
from modules.train_utils import *
import modules.constants as const
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from modules.train_utils import TrainInputs

class MatlabDataSet(object):
    def __init__(self,verbose=0, debug=0) -> None:
        
        """

        Args:
            verbose (int, optional): [description]. Defaults to 0.
        """
        super().__init__()
        self.type= 'MatlabDataSet'
        self.verbose= verbose
        self.debug= debug
        self.dataset=dict()
        self.fwd_model=dict()
        self.user_entry=dict()
        self.samples = dict()
        self.path_pkl= ''
        self.X= []
        self.Y=[]
        self.filename= ''
        self.path= ''

    def flex_load(self, path="", auto=False, type2load=const.EXT_MAT, time=None):

        if verify_file(path, extension=const.EXT_MAT):
            self.mk_dataset_from_matlab(path=path, auto= auto, time=time)
        elif verify_file(path, extension=const.EXT_PKL):
            self.load_dataset_from_pickle(path)
        else:
            if type2load==const.EXT_MAT:
                self.mk_dataset_from_matlab(auto= auto, time=time)
            else:
                self.load_dataset_from_pickle(path)
            
        


    def mk_dataset_from_matlab(self, path="", auto= False, only_get_samples_EIDORS=False, time= None):
        """[summary]

        Args:
            path (str, optional): [description]. Defaults to "".
        """
            
        if self.debug:
            self.filename, self.path ="test10_infos2py.mat", "E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/DStest"
        else:
            
            if verify_file(path, extension=const.EXT_MAT):
                self.path, self.filename= os.path.split(path)
            else:
                self.path, self.filename =get_file(filetypes=[("Matlab file","*.mat")])

        filename= os.path.join(self.path, self.filename)
        if verify_file(filename, extension=const.EXT_MAT):
            if self.verbose or 1:
                print(  '##################################################\n',\
                        'Loading file: {}\n'.format(self.filename), \
                        'path: ...{}\n'.format(self.path[-60:]),\
                        '##################################################')
            if not only_get_samples_EIDORS:
                self.get_info_from_dataset(self.filename, self.path)
                self.load_samples(auto=auto)
                self.save_dataset(time=time)
            else:
                self.load_samples_EIDORS(os.path.join(self.path, self.filename))
        else:
            print(  '##################################################\n',\
                        'Loading CANCELLED\n',\
                        '##################################################')

    def load_dataset_from_pickle(self, path=""):
        """load a MatlabDataSet from a pickle-file

        Returns:
            loaded_dataset[MatlabDataSet]: obvious
        """
        
        if verify_file(path, extension=const.EXT_PKL):
            self.path, self.filename= os.path.split(path)
        else:
            self.path, self.filename= get_file(filetypes=[("pickle file","*.pkl")])

        
        path= self.path
        filepath= os.path.join(self.path, self.filename)
        if verify_file(filepath, extension=const.EXT_PKL):
            print(  '##################################################\n',\
                    'Loading file: {}\n'.format(self.filename), \
                    'path: ...{}\n'.format(self.path[-60:]),\
                    '##################################################')

            self= load_pickle(filepath, class2upload=self)
            self.path_pkl=filepath # as we do not save the pickel we have to actualizate the path (win/unix)
            self.path= path # we have to actualizate the path (win/unix)
            self.load_samples(mode='reload')
            #self.save_dataset()
        else:
            print(  '##################################################\n',\
                    'Loading CANCELLED\n',\
                    '##################################################')


    def get_info_from_dataset(self, filename, path):
        """ extract the data  contained in the *info2py.mat to load the samples in python.

        Args:
            filename (str): mat-file ending with *info2py.mat
            path (str): folder where the mat-file is to found
        """

        file = loadmat(os.path.join(path, filename))
        #sort the data using the keys name
        for key in file.keys():
            if ("userentry") in key:
                keynew= key.replace("userentry_", "")
                if ("fmdl") in key:
                    keynew= keynew.replace("fmdl_", "")   
                    self.fwd_model[keynew]= file[key]
                else:
                    self.user_entry[keynew]= file[key]
            else:
                if ("__") not in key:
                    self.dataset[key]= file[key]

        # Samples folder /filenames extract
        self.dataset["samplesfolder"]= self.str_cellarray2str_list(self.dataset["samplesfolder"])
        self.dataset["samplesfilenames"]= self.str_cellarray2str_list(self.dataset["samplesfilenames"])
        self.dataset["samplesindx"]= self.dataset["samplesindx"]
        if self.debug:
            print('\nKeys of loaded mat file:', file.keys())
        if self.verbose:
            print('\nKeys of dataset:',self.dataset.keys())
            print('\nKeys of fwd_model:',self.fwd_model.keys())
            print('\nKeys of user_entry:',self.user_entry.keys())
           
    def str_cellarray2str_list(self, str_cellarray):
        """ After using loadmat, the str cell array have a strange shape
            >>> here the loaded "strange" array is converted to an str list

        Args:
            str_cellarray ("strange" ndarray): correponing to str cell array in matlab

        Returns:
            str list: 
        """
        if str_cellarray.ndim ==2: 
            tmp= str_cellarray[0,:]
            str_array= [ t[0] for t in tmp] 
        elif str_cellarray.ndim ==1:
            tmp= str_cellarray[0]
            str_array= [tmp] 

        return str_array 

    def get_number_samples2load(self,number_samples2load=0, auto= False):
        """ Get nb of samples to load (console input from user)

        Args:
            number_samples2load (int, optional):if > 0 console input wont be asked. Defaults to 0.

        Returns:
            number_samples2load [int]: nb of samples to load
        """
        number_samples2load= np.amax(self.dataset["samplesindx"])
        if not number_samples2load or not auto: 
            prompt= "\n{} samples are availables. \nEnter the number of samples to load (Enter for all): \n".format(number_samples2load)
            input_user=input(prompt)
            try:
                number_samples2load = int(input_user)
            except ValueError:
                pass
        if self.verbose:
            print('\nNumber of samples to load : {}'.format(number_samples2load))
        return number_samples2load

    def get_keys_of_samples(self, path="", auto= False, keys_default= ['X','y']):
        """ set the keys to load of the samples ()

        Args:
            
            path (str, optional): path of a sample file to get the list of available keys list.
                                Defaults to "". if not given default or given keys wilt be used.

        Returns:
            keys2load [str list]: [description]
        """
        
        # Deprecated we dont need that

        # if not verify_file(path, extension=".mat") and not auto:
        #     keys2load= keys_default
        # elif verify_file(path, extension=".mat") and not auto:
        #     batch_file=loadmat(path)
        #     keys= [ key  for key in batch_file.keys() if "__" not in key]
        #     input_valid= False
        #     while not input_valid:
        #         prompt= "Enter keys list (comma separated) contained in the list {} (Enter for all): \n".format(keys)
        #         input_user=input(prompt)
        #         if input_user=="":
        #             # print('Enter pressed')
        #             keys2load= keys
        #             break
        #         keys2load=input_user.split(sep=',')
        #         input_valid= True
        #         for k in keys2load:
        #             if k not in keys:
        #                 # print('Enter key contained in the list {}'.format(keys))
        #                 input_valid= False
        #                 break
        # else: # auto==true
        #     folder=os.path.join(self.path, self.dataset["samplesfolder"][0])
        #     filesnames= self.dataset["samplesfilenames"]
        #     batch_file=loadmat(os.path.join(folder, filesnames[0]),)
        #     keys2load= [ key  for key in batch_file.keys() if "__" not in key]

        
        keys2load= ['X','y']
        folder=os.path.join(self.path, self.dataset["samplesfolder"][0])
        filesnames= self.dataset["samplesfilenames"]
        batch_file=loadmat(os.path.join(folder, filesnames[0]),)
        keys2load_frombatchfile= [ key  for key in batch_file.keys() if "__" not in key]

        if not keys2load_frombatchfile==keys2load:
            error('Samples file does not contain {} variables as expected'.format(keys2load))

        if self.verbose:
            print('\nVariables of samples to load : {}'.format(keys2load))

        return  keys2load

    def load_samples(self, mode= 'load', auto=False):
        """ load the samples from each mat-file

        Args:
            mode (str, optional): 'load' or 'reload'. Defaults to 'load'. 
                                load mode ist the std one where number and key to load are aske to the user
                                relaod is used after loading a Matlabdataset from a pickle-file 

        """
        folder=os.path.join(self.path, self.dataset["samplesfolder"][0])
        filesnames= self.dataset["samplesfilenames"]

        if mode=='load':
            self.nb_samples=self.get_number_samples2load(auto=auto)
            self.keys2load= self.get_keys_of_samples(auto=auto, path=os.path.join(folder, filesnames[0]))
        elif mode == 'reload':
            if self.verbose:
                print('{} samples are reloaded, with keys {}'.format(self.nb_samples,self.keys2load))
        tmp= np.where(self.dataset["samplesindx"]==self.nb_samples)
        nb_samples_batch2load= tmp[0][0]
        nbsamples_lastbatch2load= tmp[1][0]

        for key in self.keys2load:
            self.samples[key]= np.array([])
        
        for idx_batch in range(nb_samples_batch2load+1):
            batch_filename= os.path.join(folder, filesnames[idx_batch])
            if self.verbose:
                print('\nLoading samples file : ...{}'.format(batch_filename[-50:]))
            batch_file=loadmat(batch_filename)
            if idx_batch==nb_samples_batch2load:
                for key in self.keys2load:
                    s= [slice(None)]*batch_file[key].ndim
                    s[1]= slice(0,nbsamples_lastbatch2load+1)                    
                    self.samples[key]=np.append(self.samples[key],batch_file[key][tuple(s)],axis=1)
            elif idx_batch==0:
                for key in self.keys2load:
                    self.samples[key]=batch_file[key]
            else:
                for key in self.keys2load:
                    self.samples[key]=np.append(self.samples[key],batch_file[key],axis=1)
        if self.verbose:
            for key in self.keys2load:
                print('\nSize of sample loaded ', key ,self.samples[key].shape)

    def load_samples_EIDORS(self, path):
        """ load the samples from each mat-file

        Args:
            mode (str, optional): 'load' or 'reload'. Defaults to 'load'. 
                                load mode ist the std one where number and key to load are aske to the user
                                relaod is used after loading a Matlabdataset from a pickle-file 

        """
        self.samples_EIDORS=dict()
        if verify_file(path, extension=".mat"):
            file = loadmat(path)
            for key in file.keys():
                if ("__") not in key:
                    self.samples_EIDORS[key]= file[key]
        
        print('\nkeys ', self.samples_EIDORS.keys())

    def save_dataset(self, time= None):
        """ save the MatlabDataSet under a pickle-file
                (samples are cleared to avoid a big file)
        """
        time = time if time else get_date_time()
        filename= os.path.join(self.path, f'{time}{const.EXT_PKL}')
        tmp= self.samples    
        self.samples= dict() # clear that too big if not... wil be reloaded...
        save_as_pickle(filename,self)
        self.path_pkl= get_POSIX_path(filename)
        self.samples = tmp

        
        
    def data_selection(self, data_sel= ['Xih','Yih']):
         # data selection
        tmpX= dict()
        tmpY= dict()
        
        tmpX['Xh'] = self.samples['X'][:,:,0]
        tmpY['Yh'] = self.samples['y'][:,:,0]

        tmpX['Xih'] = self.samples['X'][:,:,1]
        tmpY['Yih'] = self.samples['y'][:,:,1]

        tmpX['Xhn'] =self.samples['X'][:,:,2]
        tmpX['Xihn']=self.samples['X'][:,:,3]

        # here we can create the differences
        tmpX['Xih-Xh']= tmpX['Xih']-tmpX['Xh']
        tmpY['Yih-Yh']= tmpY['Yih']-tmpY['Yh']

        tmpX['Xihn-Xhn']= tmpX['Xihn']-tmpX['Xhn']
        tmpX['Xihn-Xh']= tmpX['Xihn']-tmpX['Xh']
        tmpX['Xihn-Xh']= tmpX['Xih']-tmpX['Xhn']

        #tri = np.array(self.fwd_model['elems'])
        #pts = np.array(self.fwd_model['nodes'])

        # perm= fwd_model['un2']    
        #perm= np.reshape(perm, (perm.shape[0],))

        #tri = tri-1 # matlab count from 1 python from 0
        #tri= interp2d.check_order(pts, tri)
        #data=dict()

        #data_node= interp2d.pts2sim(pts, tri, perm)

        ## control input.... TODO

        if data_sel[0] not in tmpX.keys():
            error('\n not correct data_sel')
        if data_sel[1] not in tmpY.keys():
            error('\n not correct data_sel')

        self.data_sel= data_sel        

        if self.verbose:
            print('\nData {} used'.format(data_sel))

        self.X= tmpX[data_sel[0]]
        self.Y= tmpY[data_sel[1]]

        return self.X, self.Y


def get_XY_from_MalabDataSet(path="", data_sel= ['Xih','Yih'], verbose=False,**kwargs):
    """[summary]

    Args:
        path (str, optional): [description]. Defaults to "".
        data_sel (list, optional): [description]. Defaults to ['Xih','Yih'].
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    raw_data=MatlabDataSet(verbose=verbose)
    raw_data.flex_load(path, **kwargs)
    raw_data.data_selection(data_sel=data_sel)
    return raw_data

def save_idx_samples_2matfile(raw_data, training_dataset, time= None):
    """[summary]

    Args:
        raw_data ([type]): [description]
        training_dataset ([type]): [description]
        feature_or_label_indexes (int, optional): [description]. Defaults to 2.
    """

    f=dict()
    f['idx_train']= training_dataset.idx_train
    f['idx_val']= training_dataset.idx_val
    f['idx_test']= training_dataset.idx_test
    time = time if time else get_date_time()

    path =  os.path.join(raw_data.path, f'{time}{const.EXT_IDX_FILE}')
    savemat(path, f, appendmat=True)
    return path


class FeaturesLabelsSet(object):
    def __init__(self) -> None:
        super().__init__()
        self.type= 'FeaturesLabelsSet'
        self.features=np.array([])
        self.labels = np.array([])
        pass
    def set_data(self, features, labels):
        self.features=features
        self.labels = labels


class EITDataset4ML(object):
    def __init__(self,verbose=False) -> None:
        super().__init__()
        self.type= 'EITDataset4ML'
        self.use_tf_dataset= False
        self.nb_samples =  0
        self.batch_size = 32
        self.test_ratio= 0.20
        self.val_ratio =0.2
        self.train=[]
        self.val=[]
        self.test=[]
        self.train_len=[]
        self.val_len=[]
        self.test_len= []
        self.verbose= verbose
        self.fwd_model=dict()
        self.src_file= ''
        # self.use_tf_dataset= use_tf_dataset
        # if self.use_tf_dataset:
        #     self.train=tf.data.Dataset()
        #     self.val=tf.data.Dataset()
        #     self.test=tf.data.Dataset()
        # else:
        #     self.train_X=FeaturesLabelsSet()
        #     self.val_X=FeaturesLabelsSet()
        #     self.test_X=FeaturesLabelsSet()
        pass
    def set_sizes_dataset(self, X, Y, batch_size = 32, test_ratio= 0.20, val_ratio=0.20):
        
        if test_ratio+val_ratio>=0.8:
            test_ratio= 0.2
            val_ratio=0.2
        
        self.nb_samples= np.shape(X)[0]
        self.batch_size = batch_size
        self.test_ratio= test_ratio
        self.val_ratio = val_ratio
        self.features_size= np.shape(X)[1]
        self.labels_size= np.shape(Y)[1]

        # if self.verbose:
        #     print(self.train_len)
        #     print(self.val_len )
        #     print(self.test_len )
    
    def mk_std_dataset(self, X, Y, batch_size = 32, test_ratio= 0.20, val_ratio=0.20, train_inputs:TrainInputs=None):
        self.use_tf_dataset= False
        self.set_sizes_dataset(X, Y, batch_size, test_ratio, val_ratio)

        
   
        scaler = MinMaxScaler()
        # transform data
        X=scaler.fit_transform(X)
        Y=scaler.fit_transform(Y)
        
        #add indexes
        idx=np.reshape(range(X.shape[0]),(X.shape[0],1))
        X= np.concatenate(( X, idx ), axis=1)

        x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y,test_size=self.test_ratio)
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp, test_size=self.val_ratio)
        
        self.idx_train= x_train[:,-1]
        self.idx_val= x_val[:,-1]
        self.idx_test= x_test[:,-1]

        self.train=FeaturesLabelsSet()
        self.val=FeaturesLabelsSet()
        self.test=FeaturesLabelsSet()
        self.train.set_data(features=x_train[:,:-1], labels=y_train)
        self.val.set_data(features=x_val[:,:-1], labels=y_val)
        self.test.set_data(features=x_test[:,:-1], labels=y_test)
        # To do
        # print('\n\nATTENTION self.train_len=[] / self.val_len=[] / self.test_len= []\n\n')
        
        self.train_len=x_train.shape[0]
        self.val_len=x_val.shape[0]
        self.test_len= x_test.shape[0]

        
        if self.verbose:
            print('\nLength of train', self.train_len)
            print('Length of val',self.val_len )
            print('Length of test',self.test_len )

    def mk_std_dataset_from_idx(self, X, Y, batch_size = 32, test_ratio= 0.20, val_ratio=0.20, train_inputs:TrainInputs=None, idx:List[Any]=[]):
        self.use_tf_dataset= False
        self.set_sizes_dataset(X, Y, batch_size, test_ratio, val_ratio)

        scaler = MinMaxScaler()
        # transform data
        X=scaler.fit_transform(X)
        Y=scaler.fit_transform(Y)
        
        #add indexes
                
        self.idx_train= idx[0]
        self.idx_val= idx[1]
        self.idx_test= idx[2]

        self.train=FeaturesLabelsSet()
        self.val=FeaturesLabelsSet()
        self.test=FeaturesLabelsSet()
        self.train.set_data(features=X[self.idx_train,:], labels=Y[self.idx_train,:])
        self.val.set_data(features=X[self.idx_val,:], labels=Y[self.idx_val,:])
        self.test.set_data(features=X[self.idx_test,:], labels=Y[self.idx_test,:])
        # To do
        # print('\n\nATTENTION self.train_len=[] / self.val_len=[] / self.test_len= []\n\n')
        
        self.train_len=self.idx_train.shape[0]
        self.val_len=self.idx_val.shape[0]
        self.test_len= self.idx_test.shape[0]

        
        if self.verbose:
            print('\nLength of train', self.train_len)
            print('Length of val',self.val_len )
            print('Length of test',self.test_len )

    def mk_tf_dataset(self, X, Y, batch_size = 32, test_ratio= 0.20, val_ratio=0.20, train_inputs:TrainInputs=None):
        self.use_tf_dataset= True       
        self.set_sizes_dataset(X, Y, batch_size, test_ratio, val_ratio)

        real_data= tf.data.Dataset.from_tensor_slices((X, Y))
        indexes = tf.data.Dataset.from_tensor_slices(tf.range(self.nb_samples))
        samples = tf.data.Dataset.zip((real_data, indexes))
        #samples=samples.shuffle()


        self.train_len=int((1-self.test_ratio-self.val_ratio)*self.nb_samples)
        self.val_len=int(self.val_ratio*self.nb_samples)
        self.test_len= int(self.test_ratio*self.nb_samples)

        train_tmp= samples.take(self.train_len)
        val_test_tmp= samples.skip(self.train_len)
        val_tmp=  val_test_tmp.take(self.val_len)
        test_tmp=  val_test_tmp.skip(self.val_len)

        idx=train_tmp.map(lambda xy, idx: idx)
        self.idx_train= np.array(list(idx.as_numpy_iterator()))
        train_tmp=train_tmp.map(lambda xy, idx: xy)
        
        idx=val_tmp.map(lambda xy, idx: idx)
        self.idx_val= np.array(list(idx.as_numpy_iterator()))
        val_tmp=val_tmp.map(lambda xy, idx: xy)

        idx=test_tmp.map(lambda xy, idx: idx)
        self.idx_test= np.array(list(idx.as_numpy_iterator()))
        test_tmp=test_tmp.map(lambda xy, idx: xy)

        scaler = MinMaxScaler()

        # transform data
        train_tmp=train_tmp.map(lambda x, y: (scaler.fit_transform(x), scaler.fit_transform(y)))
        val_tmp=val_tmp.map(lambda x, y: (scaler.fit_transform(x), scaler.fit_transform(y)))
        val_tmp=val_tmp.map(lambda x, y: (scaler.fit_transform(x), scaler.fit_transform(y)))
        
        if self.verbose:
            print('\nLength of train', self.train_len)
            print('Length of val',self.val_len )
            print('Length of test',self.test_len )

        if self.batch_size:
            self.train= train_tmp.repeat().batch(self.batch_size)
            self.val= val_tmp.repeat().batch(self.batch_size)
            self.test=test_tmp.repeat().batch(self.batch_size)
        else:
            self.train= train_tmp
            self.val= val_tmp
            self.test=test_tmp
        
    
    # def get_sample(self, which= 'train', idx_samples= None):

    #     if which=='train':
    #         tmp_ds= self.train
    #         tmp_len= self.train_len
    #     elif which== 'val':
    #         tmp_ds= self.val
    #         tmp_len= self.val_len
    #     elif which== 'test':
    #         tmp_ds= self.test
    #         tmp_len= self.test_len

    #     if not idx_samples:
    #         idx_samples= np.random.randint(tmp_len)
    #     # To do
    #     if self.use_tf_dataset:
    #         # extract data for verification?
    #         for inputs, outputs in tmp_ds.as_numpy_iterator():
    #             if self.batch_size:
    #                 for 

    #             else:

    #             if 
    #             break
    #     else:
    #         x, y= tmp_ds.labels[idx_samples],tmp_ds.features[idx_samples]

    #     if self.verbose:
    #         print(f'Features from dataset {which}, idx {idx_samples}: {x}, {x.shape}')
    #         print(f'Labels from dataset {which}, idx {idx_samples}: {y}, {y.shape}')

    #     return x, y

# def minmax_scale(x,y, axis=0):
#     X= (x – min) / (max – min)
#     Y

#     return (x – min) / (max – min)
def scale_prepocess(x, scale):
    scaler = MinMaxScaler()
    if scale:
        x= scaler.fit_transform(x)
    return x

def dataloader( raw_data,
                batch_size = 32, 
                test_ratio= 0.20, 
                val_ratio=0.20, 
                use_tf_dataset=True, 
                verbose=False, 
                normalize=[True, True], 
                train_inputs:TrainInputs=None, 
                idx:List[Any]=[]):
    """[summary]

    Args:
        raw_data ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 32.
        test_ratio (float, optional): [description]. Defaults to 0.20.
        val_ratio (float, optional): [description]. Defaults to 0.20.
        use_tf_dataset (bool, optional): [description]. Defaults to True.
        verbose (bool, optional): [description]. Defaults to False.
        normalize (list, optional): [description]. Defaults to [True, True].

    Returns:
        [type]: [description]
    """

    if train_inputs:
        batch_size=train_inputs.batch_size
        test_ratio=train_inputs.test_ratio
        val_ratio=train_inputs.val_ratio
        use_tf_dataset=train_inputs.use_tf_dataset
        normalize= train_inputs.normalize

    # data transformation
    X = raw_data.X.T
    Y = raw_data.Y.T

    # if normalize[0]:
    #     X = tf.keras.utils.normalize(X, axis=0).astype("float32")
    # if normalize[1]:
    #     Y = tf.keras.utils.normalize(Y, axis=0).astype("float32")
 
    print('Shape of X: {},\nShape of Y: {}'.format(np.shape(X) ,np.shape(Y)))
    
    # make the training dataset 

    training_dataset= EITDataset4ML(verbose=verbose)
    training_dataset.src_file= raw_data.path_pkl
    training_dataset.fwd_model= raw_data.fwd_model

    if idx:
        training_dataset.mk_std_dataset_from_idx(X, Y, batch_size=batch_size, test_ratio=test_ratio, val_ratio=val_ratio, train_inputs= train_inputs, idx=idx)
        return training_dataset
    if use_tf_dataset:
        training_dataset.mk_tf_dataset(X, Y, batch_size=batch_size, test_ratio=test_ratio, val_ratio=val_ratio, train_inputs= train_inputs)
    else:
        training_dataset.mk_std_dataset(X, Y, batch_size=batch_size, test_ratio=test_ratio, val_ratio=val_ratio, train_inputs= train_inputs)
    
    # Reserve num_val_samples samples for validation
    return training_dataset
    
def extract_samples(dataset, dataset_part='test', idx_samples=None, elem_idx = 0):
    if dataset.use_tf_dataset:
       
        x= []
        y=[]
        for i, xy in enumerate(getattr(dataset, dataset_part)):
            # print('#', i, eval_dataset.batch_size,eval_dataset.test_len)
            if dataset.batch_size:
                if (i+1)*dataset.batch_size>dataset.test_len:
                    break
                x.append(xy[0].numpy())
                y.append(xy[1].numpy())
            else:
                #print(xy[elem_idx], xy[elem_idx].shape)
                x.append(xy[0].numpy().reshape(xy[0].shape[0],1).T)
                y.append(xy[1].numpy().reshape(xy[1].shape[0],1).T)
                
        
        samples_x = np.concatenate(x, axis=0)
        samples_y = np.concatenate(y, axis=0)
        # samples = np.concatenate(l, axis=1).T
             
    else:
        samples_x= getattr(getattr(dataset, dataset_part),'features')
        samples_y= getattr(getattr(dataset, dataset_part),'labels')

    if not idx_samples:
            idx_samples= np.random.randint(len(samples_x))
            
    if idx_samples=='all':
        return samples_x, samples_y

    if isinstance(idx_samples, int):
        idx_samples= [idx_samples]

    if isinstance(idx_samples, list):
            samples_x= samples_x[idx_samples]  
            samples_y= samples_y[idx_samples]  
              
    return samples_x, samples_y
if __name__ == "__main__":
    path= "E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/DStest/test10_infos2py.mat" 
    
    raw_data= get_XY_from_MalabDataSet(path= path, data_sel= ['Xh','Yh'])
    
    training_dataset=dataloader(raw_data, verbose=True, batch_size=1)
    for inputs, indexes in training_dataset.train.as_numpy_iterator():
            print(inputs,'indexes', indexes)
            # Print the first element and the label
            print(inputs[0])
            print('label of this input is', inputs[1])
            break
    pass

