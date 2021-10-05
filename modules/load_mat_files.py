import numpy as np
from scipy.io import loadmat
from scipy.io.matlab.mio import savemat
import tensorflow as tf
import os
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import pickle
from modules.path_utils import *

class MatlabDataSet():

    def __init__(self, verbose=0, debug=0) -> None:
        """

        Args:
            verbose (int, optional): [description]. Defaults to 0.
        """
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

    def flex_load(self, path="", auto=False, type2load='.mat'):
        if verify_file(path, extension=".mat"):
            self.mk_dataset_from_matlab(path=path, auto= auto)
        elif verify_file(path, extension=".pkl"):
            self.load_dataset_from_pickle(path)
        else:
            if type2load=='.mat':
                self.mk_dataset_from_matlab(auto= auto)
            else:
                self.load_dataset_from_pickle(path)
            
        


    def mk_dataset_from_matlab(self, path="", auto= False, only_get_samples_EIDORS=False):
        """[summary]

        Args:
            path (str, optional): [description]. Defaults to "".
        """
            
        if self.debug:
            self.filename, self.path ="test10_infos2py.mat", "E:/EIT_Project/05_Engineering/04_Software/Python/eit_tf_workspace/datasets/DStest"
        else:
            
            if verify_file(path, extension=".mat"):
                self.path, self.filename= os.path.split(path)
            else:
                self.path, self.filename =self.get_file()

        filename= os.path.join(self.path, self.filename)
        if verify_file(filename, extension=".mat"):
            if self.verbose or 1:
                print(  '##################################################\n',\
                        'Loading file: {}\n'.format(self.filename), \
                        'path: ...{}\n'.format(self.path[-60:]),\
                        '##################################################')
            if not only_get_samples_EIDORS:
                self.get_info_from_dataset(self.filename, self.path)
                self.load_samples(auto=auto)
                self.save_dataset()
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
        
        if verify_file(path, extension=".pkl"):
            self.path, self.filename= os.path.split(path)
        else:
            self.path, self.filename= self.get_file(filetypes=[("pickle file","*.pkl")])

        filename= os.path.join(self.path, self.filename)
        if verify_file(filename, extension=".pkl"):
            if self.verbose or 1:
                print(  '##################################################\n',\
                        'Loading file: {}\n'.format(self.filename), \
                        'path: ...{}\n'.format(self.path[-60:]),\
                        '##################################################')

            
            with open(filename, 'rb') as inp:
                loaded_dataset = pickle.load(inp)

            for key in loaded_dataset.__dict__.keys():
                setattr(self, key, getattr(loaded_dataset,key))
            self.load_samples(mode='reload')
            self.save_dataset()
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

    def get_file(self, filetypes=[("Matlab file","*.mat")]):
        """used to get select files using gui (multple type of file cane be set!)

        Args:
            filetypes (list, optional: obvious.... Defaults to [("Matlab file","*.mat")].

        Returns:
            filename (str): filename of the file selected
            path (str): folder where the mat-file is to found
        """
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        whole_path = askopenfilename(   initialdir=os.getcwd(),
                                        filetypes=filetypes) # show an "Open" dialog box and return the path to the selected file
        path, filename = os.path.split(whole_path)
        if self.verbose:
            print(path, filename)
        return path, filename

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
        

        if not verify_file(path, extension=".mat") and not auto:
            keys2load= keys_default
        elif verify_file(path, extension=".mat") and not auto:
            batch_file=loadmat(path)
            keys= [ key  for key in batch_file.keys() if "__" not in key]
            input_valid= False
            while not input_valid:
                prompt= "Enter keys list (comma separated) contained in the list {} (Enter for all): \n".format(keys)
                input_user=input(prompt)
                if input_user=="":
                    # print('Enter pressed')
                    keys2load= keys
                    break
                keys2load=input_user.split(sep=',')
                input_valid= True
                for k in keys2load:
                    if k not in keys:
                        # print('Enter key contained in the list {}'.format(keys))
                        input_valid= False
                        break
        else: # auto==true
            folder=os.path.join(self.path, self.dataset["samplesfolder"][0])
            filesnames= self.dataset["samplesfilenames"]
            batch_file=loadmat(os.path.join(folder, filesnames[0]),)
            keys2load= [ key  for key in batch_file.keys() if "__" not in key]

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

        


        # folder=os.path.join(self.path, self.dataset["samplesfolder"][0])
        # filesnames= self.dataset["samplesfilenames"]

        # if mode=='load':
        #     self.nb_samples=self.get_number_samples2load(auto=auto)
        #     self.keys2load= self.get_keys_of_samples(auto=auto, path=os.path.join(folder, filesnames[0]))
        # elif mode == 'reload':
        #     if self.verbose:
        #         print('{} samples are reloaded, with keys {}'.format(self.nb_samples,self.keys2load))
        # tmp= np.where(self.dataset["samplesindx"]==self.nb_samples)
        # nb_samples_batch2load= tmp[0][0]
        # nbsamples_lastbatch2load= tmp[1][0]

        # for key in self.keys2load:
        #     self.samples[key]= np.array([])
        
        # for idx_batch in range(nb_samples_batch2load+1):
        #     batch_filename= os.path.join(folder, filesnames[idx_batch])
        #     if self.verbose:
        #         print('\nLoading samples file : ...{}'.format(batch_filename[-50:]))
        #     batch_file=loadmat(batch_filename)
        #     if idx_batch==nb_samples_batch2load:
        #         for key in self.keys2load:
        #             s= [slice(None)]*batch_file[key].ndim
        #             s[1]= slice(0,nbsamples_lastbatch2load+1)                    
        #             self.samples[key]=np.append(self.samples[key],batch_file[key][tuple(s)],axis=1)
        #     elif idx_batch==0:
        #         for key in self.keys2load:
        #             self.samples[key]=batch_file[key]
        #     else:
        #         for key in self.keys2load:
        #             self.samples[key]=np.append(self.samples[key],batch_file[key],axis=1)
        # if self.verbose:
        #     for key in self.keys2load:
        #         print('\nSize of sample loaded ', key ,self.samples[key].shape)


    def save_dataset(self):
        """ save the MatlabDataSet under a pickle-file
                (samples are cleared to avoid a big file)
        """
        filename= os.path.join(self.path, self.filename.replace(".mat", ".pkl"))
        tmp= self.samples    
        self.samples= dict() # clear that too big if not... wil be reloaded...
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        self.path_pkl= filename   
        self.samples = tmp

        if self.verbose:
            print('\nMatlabDataSet saved in : ...{}'.format(filename[-50:]))
        

    # def verify_file(self, path, extension):
    #     path_out=""
    #     if os.path.isfile(path):
    #             _, file_extension = os.path.splitext(path)
    #             if file_extension==extension:
    #                 path_out= path
    #     return path_out

def get_XY_from_MalabDataSet(path="", data_sel= ['Xih','Yih'], verbose=False,**kwargs):

    raw_data=MatlabDataSet(verbose=verbose)
    raw_data.flex_load(path, **kwargs)
    raw_data.fwd_model

    # data selection
    tmp= dict()

    tmp['Xh'] = raw_data.samples['X'][:,:,0]
    tmp['Yh'] = raw_data.samples['y'][:,:,0]
    # print(raw_data.samples['y'][:,0,0])
    # print(raw_data.samples['y'][:,0,1])
    tmp['Xih'] = raw_data.samples['X'][:,:,1]
    tmp['Yih'] = raw_data.samples['y'][:,:,1]

    tmp['Xhn'] =raw_data.samples['X'][:,:,2]
    tmp['Xihn']=raw_data.samples['X'][:,:,3]

    
    data_sel_tmp= ['Xih','Yih']
    if len(data_sel)==2:
        for key in data_sel:
            if key not in tmp.keys():
                data_sel_tmp= ['Xih','Yih']
                print('\n not correct data_sel')
                break
            else:
                data_sel_tmp= data_sel

    print('\nData {} used'.format(data_sel_tmp))

    raw_data.X= tmp[data_sel_tmp[0]]
    raw_data.Y= tmp[data_sel_tmp[1]]

    return raw_data

def save_2matfile_4_solving(raw_data, training_dataset, feature_or_label_indexes=2):
    
    # l=[]
    # for i, xy in enumerate(training_dataset.test):
    #     print('#', i, training_dataset.batch_size,training_dataset.test_len) 
    #     if (i+1)*training_dataset.batch_size>training_dataset.test_len:
    #         break
    #     l.append(xy[feature_or_label_indexes].numpy())
            
    # x = np.concatenate(l, axis=0)

    f=dict()
    f['idx_train']= training_dataset.idx_train
    f['idx_val']= training_dataset.idx_val
    f['idx_test']= training_dataset.idx_test
    path =  os.path.join(raw_data.path, 'testdataset_U4Solve.mat')
    savemat(path, f, appendmat=True)

    
if __name__ == "__main__":
     a= MatlabDataSet(verbose=True)
     a.mk_dataset_from_matlab(only_get_samples_EIDORS=True)
     

    # b=MatlabDataSet()
    # b= b.load_dataset_from_pickle()
   

    # for key in a.dataset.keys():
    #     print(key, ' : ', a.dataset[key], type(a.dataset[key]))
    # for key in a.fwd_model.keys():
    #     print(key, ' : ', a.fwd_model[key], type(a.fwd_model[key]))
    # filename, path = get_mat_file()

    # Xih = loadmat()
    # print(type(Xih['samplesfilenames']))
    # print(Xih['samplesfilenames'][0,:][1][0])
    # Xih = loadmat(r".\\datasets\\DStest\\Samples\\{}".format(Xih['samplesfilenames'][0,:][1][0]))
    # print(Xih.keys())