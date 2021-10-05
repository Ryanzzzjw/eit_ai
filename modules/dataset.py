
import numpy as np
import tensorflow as tf
from modules.load_mat_files import *
# from load_mat_files import *
import sklearn.model_selection

class FeaturesLabelsSet():
    def __init__(self) -> None:
        self.features=np.array([])
        self.labels = np.array([])
        pass
    def set_data(self, features, labels):
        self.features=features
        self.labels = labels


class EITDataset4ML():
    def __init__(self, verbose=False) -> None:
        self.use_tf_dataset= False
        self.nb_samples =  0
        self.batch_size = 32
        self.test_size= 0.20
        self.val_size =0.2
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
    def set_sizes_dataset(self, X, Y, batch_size = 32, test_size= 0.20, val_size=0.20):
        
        if test_size+val_size>=0.8:
            test_size= 0.2
            val_size=0.2
        
        self.nb_samples= np.shape(X)[0]
        self.batch_size = batch_size
        self.test_size= test_size
        self.val_size = val_size
        self.features_size= np.shape(X)[1]
        self.labels_size= np.shape(Y)[1]

        # if self.verbose:
        #     print(self.train_len)
        #     print(self.val_len )
        #     print(self.test_len )
    
    def mk_std_dataset(self,X, Y, batch_size = 32, test_size= 0.20, val_size=0.20):
        self.use_tf_dataset= False
        self.set_sizes_dataset(X, Y, batch_size, test_size, val_size)

        x_tmp, x_test, y_tmp, y_test = sklearn.model_selection.train_test_split(X, Y,test_size=self.test_size)
        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_tmp, y_tmp, test_size=self.val_size)
        
        self.train=FeaturesLabelsSet()
        self.val=FeaturesLabelsSet()
        self.test=FeaturesLabelsSet()
        self.train.set_data(features=x_train, labels=y_train)
        self.val.set_data(features=x_val, labels=y_val)
        self.test.set_data(features=x_test, labels=y_test)
        # To do
        # print('\n\nATTENTION self.train_len=[] / self.val_len=[] / self.test_len= []\n\n')
        
        self.train_len=x_train.shape[0]
        self.val_len=x_val.shape[0]
        self.test_len= x_test.shape[0]
        
        if self.verbose:
            print('\nLength of train', self.train_len)
            print('Length of val',self.val_len )
            print('Length of test',self.test_len )

    def mk_tf_dataset(self, X, Y, batch_size = 32, test_size= 0.20, val_size=0.20):
        self.use_tf_dataset= True       
        self.set_sizes_dataset(X, Y, batch_size, test_size, val_size)

        real_data= tf.data.Dataset.from_tensor_slices((X, Y))
        indexes = tf.data.Dataset.from_tensor_slices(tf.range(self.nb_samples))
        samples = tf.data.Dataset.zip((real_data, indexes))
        #samples=samples.shuffle()


        self.train_len=int((1-self.test_size-self.val_size)*self.nb_samples)
        self.val_len=int(self.val_size*self.nb_samples)
        self.test_len= int(self.test_size*self.nb_samples)

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

        
        if self.verbose:
            print('\nLength of train', self.train_len)
            print('Length of val',self.val_len )
            print('Length of test',self.test_len )


        self.train= train_tmp.repeat().batch(self.batch_size)
        self.val= val_tmp.repeat().batch(self.batch_size)
        self.test=test_tmp.repeat().batch(self.batch_size)
        


    def get_sample(self):
        # To do
        pass
def dataloader(raw_data, batch_size = 32, test_size= 0.20, val_size=0.20, use_tf_dataset=True, verbose=False):

    #def dataloader(path="", data_sel= ['Xih','Yih'], batch_size = 32, test_size= 0.20, val_size=0.20, use_tf_dataset=True, verbose=False,**kwargs):
    # ###########################################
    # # PUT IN Load_mat_file
    # # data loading
    # raw_data=MatlabDataSet(verbose=verbose)
    # raw_data.flex_load(path, **kwargs)
    # raw_data.fwd_model

    # # data selection
    # tmp= dict()

    # tmp['Xh'] = raw_data.samples['X'][:,:,0]
    # tmp['Yh'] = raw_data.samples['y'][:,:,0]
    # # print(raw_data.samples['y'][:,0,0])
    # # print(raw_data.samples['y'][:,0,1])
    # tmp['Xih'] = raw_data.samples['X'][:,:,1]
    # tmp['Yih'] = raw_data.samples['y'][:,:,1]

    # tmp['Xhn'] =raw_data.samples['X'][:,:,2]
    # tmp['Xihn']=raw_data.samples['X'][:,:,3]

    
    # data_sel_tmp= ['Xih','Yih']
    # if len(data_sel)==2:
    #     for key in data_sel:
    #         if key not in tmp.keys():
    #             data_sel_tmp= ['Xih','Yih']
    #             print('\n not correct data_sel')
    #             break
    #         else:
    #             data_sel_tmp= data_sel

    # print('\nData {} used'.format(data_sel_tmp))


    # ###########################################

    # X= tmp[data_sel_tmp[0]]
    # Y= tmp[data_sel_tmp[1]]

    # data transformation
    X = raw_data.X.T
    Y = raw_data.Y.T
    X = tf.keras.utils.normalize(X, axis=0).astype("float32")
    Y = Y.astype("float32")
 
    print('Shape of X: {},\nShape of Y: {}'.format(np.shape(X) ,np.shape(Y)))
    
    # make the training dataset 

    training_dataset= EITDataset4ML(verbose=verbose)
    training_dataset.src_file= raw_data.path_pkl

    if use_tf_dataset:
        training_dataset.mk_tf_dataset(X, Y, batch_size=batch_size, test_size=test_size, val_size=val_size)
    else:
        training_dataset.mk_std_dataset(X, Y, batch_size=batch_size, test_size=test_size, val_size=val_size)
    
    training_dataset.fwd_model= raw_data.fwd_model
    # Reserve num_val_samples samples for validation
    return training_dataset

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

