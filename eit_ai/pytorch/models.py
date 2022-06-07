import os
from abc import ABC, abstractmethod
import logging
from typing import Any
from contextlib import redirect_stdout
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from eit_ai.pytorch.const import (PYTORCH_LOSS, PYTORCH_MODEL_SAVE_FOLDERNAME,
                                  PYTORCH_OPTIMIZER)
from eit_ai.pytorch.dataset import (DataloaderGenerator,
                                    StdPytorchDatasetHandler)
from eit_ai.train_utils.dataset import AiDatasetHandler
from eit_ai.train_utils.lists import (ListPyTorchLosses,
                                      ListPytorchModelHandlers,
                                      ListPytorchModels, ListPyTorchOptimizers,
                                      get_from_dict)
from eit_ai.train_utils.metadata import MetaData, reload_metadata
from eit_ai.train_utils.models import (MODEL_SUMMARY_FILENAME,
                                       AiModelHandler, ModelNotDefinedError,
                                       WrongLearnRateError, WrongMetricsError)
from genericpath import isdir, isfile
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
writer = SummaryWriter()

class TypicalPytorchModel(ABC):
    """Define a standard pytorch Model
    """    
    
    net:nn.Module=None
    name:str=None
    def __init__(self, metadata: MetaData) -> None:
        super().__init__()
        self._set_layers(metadata)
    
    @abstractmethod
    def _set_layers(self, metadata:MetaData)-> None:
        """define the layers of the model and the name

        Args:
            metadata (MetaData): [description]
        """ 
    
    def prepare(self, op:torch.optim.Optimizer, loss):
        self.optimizer= op
        self.loss= loss
        
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # logger.debug(f'foward, {x.shape=}')
        return self.net(x)


    def train_single_epoch(self, dataloader:DataLoader) -> Any:
        self.net.train()
        # logger.debug(f'run_single_epoch')
        for idx, data_i in enumerate(dataloader):
            # logger.debug(f'Batch #{idx}')
            train_loss = 0
            inputs, labels = data_i
            inputs = inputs.to(device=0)
            labels = labels.to(device=0)
            y_pred = self.net(inputs)
            #loss
            loss_value = self.loss(y_pred, labels)
            train_loss += loss_value.item()
            
            #backward propagation
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()  #update
            
            # logger.debug(f'Batch #{idx}: loss={loss_value.item():.6f}')

        # logger.info(f'loss={train_loss:.6f}\n--------------------------')
        return train_loss / len(dataloader)
    
    def val_single_epoch(self, dataloader: DataLoader) -> Any:
        # logger.debug(f'run_single_validation_epoch')
        val_loss = 0
        with torch.no_grad():
            for idx, data_i in enumerate(dataloader):
                
                inputs, labels = data_i
                inputs = inputs.to(device=0)
                labels = labels.to(device=0)
                y_pred = self.net(inputs)
                
                loss_value = self.loss(y_pred, labels)
                val_loss += loss_value.item()
        return val_loss / len(dataloader) 

    def get_name(self)->str:
        """Return the name of the model/network

        Returns:
            str: specific name of the model/network
        """        
        return self.name

    def get_net(self):
        return self.net

    def predict(self, x_pred: np.ndarray)->np.ndarray:
        """[summary]
        predict the new x
        """
        return self.net(torch.Tensor(x_pred)).detach().numpy()

class StdPytorchModel(TypicalPytorchModel):

    def _set_layers(self, metadata:MetaData)-> None:
        in_size=metadata.input_size
        out_size=metadata.output_size
        self.name= "MLP with 3 layers"
        self.net = nn.Sequential(nn.Linear(in_size,512),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(512, 1024),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 2048),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(2048, out_size),
                                nn.Sigmoid()
                                )
       
        self.net.to(device=0)

class Conv1dNet(TypicalPytorchModel):
    
    def _set_layers(self, metadata: MetaData) -> None:

        out_size=metadata.output_size
        self.name = "1d CNN"
        self.net = torch.nn.Sequential(nn.Conv1d(in_channels= 1, out_channels= 8, kernel_size=8, stride=1, padding=0),
                                       nn.ReLU(True),
                                       nn.MaxPool1d(kernel_size=2, stride=2),
                                       nn.Conv1d(8, 8, kernel_size=8, stride=1, padding=0),
                                       nn.ReLU(True),
                                       nn.MaxPool1d(kernel_size=2, stride=2),
                                       nn.Conv1d(8, 16, kernel_size=16, stride=1, padding=0),
                                       nn.ReLU(True),
                                       nn.MaxPool1d(kernel_size=2, stride=2),
                                       nn.Flatten(),
                                       nn.Linear(512, 512),
                                       nn.ReLU(True),
                                       nn.Linear(512, out_size),
                                       nn.Sigmoid()
                                    )
        self.net.to(device=0)    

class AutoEncoder(TypicalPytorchModel):  
      
    def _set_layers(self, metadata: MetaData) -> None:

        in_size = metadata.input_size
        out_size=metadata.output_size
        self.name = "AutoEncoder"
        self.net = torch.nn.Sequential()
        
        encoder = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),  
            )
        
        decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, out_size),
            nn.Sigmoid(),
            )
        
        
        self.net.add_module('encoder', encoder)
        self.net.add_module('decoder', decoder)
        
        self.net.to(device=0)
        
################################################################################
# Std PyTorch ModelManager
################################################################################
class StdPytorchModelHandler(AiModelHandler):

    def _define_model(self, metadata:MetaData)-> None:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device Cuda: {device}")

        model_cls=get_from_dict(
            metadata.model_type, PYTORCH_MODELS, ListPytorchModels)
        self.model=model_cls(metadata)
        self.name= self.model.get_name()

    def _get_specific_var(self, metadata:MetaData)-> None:   
        self.specific_var['optimizer'] = get_pytorch_optimizer(metadata, self.model.get_net())
        self.specific_var['loss'] = get_pytorch_loss(metadata)
        if not isinstance(metadata.metrics, list):
            raise WrongMetricsError(f'Wrong metrics type: {metadata.metrics}')
        self.specific_var['metrics'] = metadata.metrics   

    def _prepare_model(self)-> None:
        self.model.prepare( 
            self.specific_var['optimizer'],
            self.specific_var['loss']  )

    def train(self, dataset:AiDatasetHandler, metadata:MetaData)-> None:
        gen=DataloaderGenerator()
        train_dataloader=gen.make(dataset, 'train', metadata=metadata)
        val_dataloader=gen.make(dataset, 'val', metadata=metadata)
        logger.info(f'Training - Started {metadata.epoch}')
        for epoch in range(metadata.epoch):
            train_loss = self.model.train_single_epoch(train_dataloader)
            val_loss = self.model.val_single_epoch(val_dataloader)
            # logger.info(f'Epoch #{epoch+1}/{metadata.epoch} : {loss=}')
            logger.info(f'Epoch #{epoch+1}/{metadata.epoch}')
            logger.info(f'train_loss = {train_loss}, val_loss = {val_loss}')
            
            writer.add_scalar("training_loss", train_loss, epoch+1)
            writer.add_scalar("val_loss", val_loss, epoch+1)

            writer.close()   

    def predict(
        self,
        X_pred:np.ndarray,
        metadata:MetaData,
        **kwargs)->np.ndarray:

        # X_pred preprocess if needed
        # if X_pred.shape[0]==1:
        #     return self.model.predict(X_pred)
        # else:
        #     res = np.array([])
        #     for i in range(X_pred.shape[0]):
        #         pred = self.model.predict(X_pred[i])
        #         res = np.append(res, pred)
        #     return res  
        return self.model.predict(X_pred)

    def save(self, metadata:MetaData)-> str: 
        
        return save_pytorch_model(self.model.net, dir_path=metadata.dir_path, save_summary=metadata.save_summary)

    def load(self, metadata:MetaData)-> None:
        model_cls=get_from_dict(
        metadata.model_type, PYTORCH_MODELS, ListPytorchModels)
        
        self.model = model_cls(metadata)
        
        self.model.net= load_pytorch_model(dir_path=metadata.dir_path)
        self.model.net.eval()
        

################################################################################
# common methods
################################################################################

def assert_pytorch_model_defined(model:Any)-> nn.Module:
    """allow to react if model not  defined

    Args:
        model (Any): [description]

    Raises:
        ModelNotDefinedError: [description]

    Returns:
        pytorch.models.Model: [description]
    """    
    if not isinstance(model, nn.Module):
        raise ModelNotDefinedError(f'Model has not been correctly defined: {model}')
    return model


def get_pytorch_optimizer(metadata:MetaData, net:nn.Module)-> torch.optim.Optimizer:

    if not metadata.optimizer:
        metadata.optimizer=list(PYTORCH_OPTIMIZER.keys())[0].value

    op_cls=get_from_dict(
        metadata.optimizer, PYTORCH_OPTIMIZER, ListPyTorchOptimizers)

    if metadata.learning_rate:
        if metadata.learning_rate >= 1.0:
            raise WrongLearnRateError(f'Wrong learning rate type (>= 1.0): {metadata.learning_rate}') 
        return op_cls(net.parameters(), lr= metadata.learning_rate)
    
    logger.warning('Learningrate has been set to 0.001!!!')
    return op_cls(net.parameters(), lr=0.0001)
        

def get_pytorch_loss(metadata:MetaData)->nn.modules.loss:

    if not metadata.loss:
        metadata.loss=list(PYTORCH_LOSS.keys())[0].value

    loss_cls=get_from_dict(metadata.loss, PYTORCH_LOSS, ListPyTorchLosses)
    return loss_cls()

def save_pytorch_model(net:nn.Module, dir_path:str='', save_summary:bool=False)-> str:
    """Save a pytorch model, additionnaly can be the summary of the model be saved"""
    if not isdir(dir_path):
        dir_path=os.getcwd()
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    
    torch.save(net, model_path)

    logger.info(f'PyTorch model saved in: {model_path}')

    if save_summary:
    
        summary_path= os.path.join(dir_path, MODEL_SUMMARY_FILENAME)
        with open(summary_path, 'w') as f:
            with redirect_stdout(f):
                summary(net, input_size=(32000, 256), device='cpu')
        logger.info(f'pytorch model summary saved in: {summary_path}')
    
    return model_path

def load_pytorch_model(dir_path:str='') -> nn.Module:
    """Load pytorch Model and return it if succesful if not """
    
    metadata = reload_metadata(dir_path=dir_path)
    if not isdir(dir_path):
        logger.info(f'pytorch model loading - failed, wrong dir {dir_path}')
        return
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    if not isfile(model_path):
        logger.info(f'pytorch model loading - failed, {PYTORCH_MODEL_SAVE_FOLDERNAME} do not exist in {dir_path}')
        return None
    try:
        net = torch.load(model_path)
        logger.info(f'pytorch model loaded: {model_path}')
        logger.info('pytorch model summary:')
        if metadata.model_type == 'Conv1dNet':
            summary(net, input_size=(metadata.batch_size, 1, 256), device='cpu')
        else:
            summary(net, input_size=(metadata.batch_size, 256), device='cpu')
        return net.eval()

    except BaseException as e: 
        logger.error(f'Loading of model from dir: {model_path} - Failed'\
                     f'\n({e})')
        return None

################################################################################
# pytorch Models
################################################################################


PYTORCH_MODEL_HANDLERS={
    ListPytorchModelHandlers.PytorchModelHandler: StdPytorchModelHandler, 
}

PYTORCH_MODELS={
    ListPytorchModels.StdPytorchModel: StdPytorchModel, 
    ListPytorchModels.Conv1dNet: Conv1dNet,
    ListPytorchModels.AutoEncoder: AutoEncoder,
}


if __name__ == "__main__":
    import logging

    from eit_ai.raw_data.matlab import MatlabSamples
    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    


    X = np.random.randn(100, 4)
    Y = np.random.randn(100)
    Y = Y[:, np.newaxis]

    raw=MatlabSamples()
    raw.X=X
    raw.Y=Y
    
    # rdn_dataset = PytorchDataset(X, Y)
    md= MetaData()
    md.set_4_dataset()
    dataset = StdPytorchDatasetHandler()
    dataset.build(raw_samples=raw, metadata= md)

    new_model = StdPytorchModelHandler()
    # for epoch in range(50):
    md.set_4_model()
    new_model.train(dataset,md)
