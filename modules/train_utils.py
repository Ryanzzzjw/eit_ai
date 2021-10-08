
from logging import error
from tensorboard import program
from tensorflow.keras.callbacks import TensorBoard

import tensorflow.keras as keras
from tensorflow.python.keras.backend import learning_phase
# from dataset import *
from modules.dataset import *
# from modules.load_mat_files import *
from modules.path_utils import *

import modules.constants as const
class TrainInputs(object):
    def __init__(self) -> None:
        super().__init__()
        self.type='TrainInputs'
        self.time=None
        self.training_name= None
        self.ouput_dir= None

        self.dataset_src_file=None
        self.idx_samples_file= None
        self.model_saving_path= None

        self.data_select= None

        self.batch_size=None
        self.test_ratio=None
        self.val_ratio=None
        self.use_tf_dataset=None
        self.normalize=None

        self.model_func=None
        self.epoch=None
        self.max_trials_autokeras=None
        self._train_len=None
        self._val_len=None
        self._test_len=None
        self.input_size=None
        self.output_size=None
        self._steps_per_epoch =None
        self._validation_steps =None
        self.callbacks=None
        self.optimizer=None
        self.learning_rate= None
        self.loss= None
        self.metrics= None
    

    def init_ouput(self, training_name, append_time= True):
        self.time = get_date_time()
        self.training_name= f'{training_name}_{self.time}' if append_time else training_name
        self.ouput_dir= mk_ouput_dir(self.training_name)

    def set_values4dataloader(  self,
                                batch_size=32,
                                test_ratio=0.2,
                                val_ratio=0.2, 
                                use_tf_dataset:bool=True, 
                                normalize= [True, True]):

        self.batch_size = batch_size
        self.test_ratio=test_ratio
        self.val_ratio=val_ratio
        if test_ratio+val_ratio>=0.5:
            self.test_ratio= 0.2
            self.val_ratio=0.2
        self.use_tf_dataset=use_tf_dataset
        self.normalize=normalize     
        
    def set_values4model(   self,
                            model_func,
                            dataset,#:EITDataset4ML, 
                            epoch=10,
                            max_trials_autokeras=10, 
                            callbacks=[],
                            optimizer= keras.optimizers.Adam(),
                            learning_rate=None,
                            loss=keras.losses.CategoricalCrossentropy(),
                            metrics=[keras.metrics.Accuracy()]):
        if not self.batch_size:
            error('call first set_values4dataloader')

        self.model_func= model_func
        self.epoch= epoch
        self.max_trials_autokeras=max_trials_autokeras
        self._train_len=dataset.train_len
        self._val_len=dataset.val_len
        self._test_len=dataset.test_len
        self.input_size=dataset.features_size
        self.output_size=dataset.labels_size
        self._steps_per_epoch = self._train_len // self.batch_size if self.batch_size else None
        self._validation_steps = self._val_len // self.batch_size if self.batch_size else None
        self.callbacks=callbacks
        
        self.dataset_src_file=[  get_POSIX_path(dataset.src_file),
                                get_POSIX_path(os.path.relpath(dataset.src_file, start=self.ouput_dir)[6:])]
        
        # filename= os.path.join(self.ouput_dir,'dataset_src_file.txt') # it that necessary??
        # save_as_txt(filename,self.dataset_src_file)
        
        self.optimizer=optimizer
        self.learning_rate= learning_rate
        if learning_rate:
            self.optimizer.learning_rate= self.learning_rate 
        self.loss=loss
        if not type(metrics)==type(list()):
            error('metrics need to be a list')
        self.metrics=metrics
    
    def set_idx_samples(self, path):
        self.idx_samples_file=[  get_POSIX_path(path),
                                get_POSIX_path(os.path.relpath(path, start=self.ouput_dir)[6:])]

    def save(self, path= None):

        if self.ouput_dir:
            path= path if path else self.ouput_dir
            filename=os.path.join(path,'train_inputs')

            #print(dill.detect.badtypes(self, depth=1).keys(), self.__dict__)

            copy=TrainInputs()

            for key, val in self.__dict__.items():

                #print(key, val, type(val), hasattr(val, '__dict__'), hasattr(val, '__call__'))

                if hasattr(val, '__dict__'):
                    setattr(copy, key, type(val).__name__)
                elif isinstance(val, list):
                    l=list()
                    for elem in val:
                        if hasattr(elem, '__dict__'):
                            #print(key, elem, type(elem), hasattr(elem, '__dict__'), hasattr(elem, '__call__'))
                            l.append(type(elem).__name__)
                        else:
                            l.append(elem)
                    setattr(copy, key, l)

                else:
                    setattr(copy, key, val)

            # save_as_pickle(filename, copy) not really posible... because of different unpickable variables
            save_as_txt(filename, copy)
        
        
    def read(self, path):
        
        load_dict=read_txt(path)
        for key in load_dict.keys():
            #print(key, load_dict[key])
            if key in self.__dict__.keys():
                setattr(self,key, load_dict[key])
        
        print(self.__dict__)
        return self
    # def load(self, path=None):
        
    #     path, filename=get_file(filetypes=[("Pickle-file", "*.pkl")], path=path)

    #     filename=os.path.join(path,filename)
    #     self=load_pickle(filename , self)


def mk_callback_tensorboard(train_inputs):

    log_path= os.path.join(train_inputs.ouput_dir,const.TENSORBOARD_LOG_FOLDER)
    
    tensorboard = TensorBoard(log_dir= log_path)
    log_tensorboard(log_path)

    return tensorboard

def log_tensorboard(log_path):

    tracking_address = log_path # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"\n######################################\nTensorflow listening on {url}\n######################################\n")


if __name__ == "__main__":

    
    learning_rate=None
    if learning_rate:
    
        print(type(keras.losses.CategoricalCrossentropy()))

    pass