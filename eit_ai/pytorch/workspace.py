
import logging

import numpy as np
from eit_ai.pytorch.dataset import PYTORCH_DATASET_HANDLERS
from eit_ai.pytorch.models import (PYTORCH_MODEL_HANDLERS, PYTORCH_MODELS,
                                   StdPytorchModelHandler)
from eit_ai.raw_data.raw_samples import RawSamples
from eit_ai.train_utils.dataset import AiDatasetHandler
from eit_ai.train_utils.lists import (ListPytorchDatasetHandlers,
                                      ListPytorchModelHandlers,
                                      ListPytorchModels, ListWorkspaces,
                                      get_from_dict)
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.workspace import (AiWorkspace, WrongDatasetError,
                                          WrongSingleXError, meas_duration)

logger = logging.getLogger(__name__)

################################################################################
# Pytorch Models
################################################################################

class PyTorchWorkspace(AiWorkspace):
    """ Generator class for pytorch models """
    def select_model_dataset(
        self, 
        model_handler: ListPytorchModelHandlers = None, 
        dataset_handler: ListPytorchDatasetHandlers = None,
        model:ListPytorchModels=None,
        metadata:MetaData=None)-> None:
        
        if model_handler is None and dataset_handler is None and model is None:
            model_handler = metadata.model_handler
            dataset_handler = metadata.dataset_handler
            model=metadata.model_type

        model_h_cls,listmodobj = get_from_dict(
            model_handler, PYTORCH_MODEL_HANDLERS, ListPytorchModelHandlers, True)
        dataset_h_cls,listdataobj= get_from_dict(
            dataset_handler,PYTORCH_DATASET_HANDLERS, ListPytorchDatasetHandlers, True)
        
        _, listmodgenobj=get_from_dict(
            model, PYTORCH_MODELS, ListPytorchModels, True)

        self.model_handler = model_h_cls()
        self.dataset_handler = dataset_h_cls()
        metadata.set_model_dataset_type(
            ListWorkspaces.PyTorch, listmodobj, listdataobj, listmodgenobj)

    def build_dataset(self, raw_samples: RawSamples, metadata: MetaData) -> None:
        """"""
        self.dataset_handler.build(raw_samples, metadata)


    def build_model(self, metadata: MetaData) -> None:
        self.model_handler.build(metadata=metadata)


    def run_training(self, metadata: MetaData = None) -> None:
        _, duration = self._run_training(metadata, return_duration=True)
        metadata.set_training_duration(duration)
        logger.info(f'### Training lasted: {duration} ###')


    @meas_duration
    def _run_training(self, metadata: MetaData = None, **kwargs) -> None:
        self.model_handler.train(dataset=self.dataset_handler, metadata=metadata)


    def get_prediction(
        self,
        metadata:MetaData,
        dataset:AiDatasetHandler=None,
        single_X:np.ndarray= None,
        **kwargs)-> np.ndarray:

        logger.info('### Prediction started: ... ###')
        prediction, duration =self._get_prediction(
            metadata, dataset= dataset,
            single_X=single_X, return_duration=True, **kwargs)
        logger.info(f'### Prediction lasted: {duration} ###')
        return prediction


    @meas_duration
    def _get_prediction(
        self,
        metadata:MetaData,
        dataset:AiDatasetHandler=None,
        single_X:np.ndarray= None,
        **kwargs)-> np.ndarray:

        
        # another dataset can be here predicted (only test part)
        if dataset is not None:
            if not isinstance(dataset, type(self.dataset_handler)): 
                raise WrongDatasetError(
                    f'{dataset= } and {self.dataset_handler} dont have same type...')
            X_pred=dataset.get_X('test')
        # Single passed X can be here predicted, after been formated
        elif single_X is not None:
            if not isinstance(single_X, np.ndarray):
                raise WrongSingleXError(f'{single_X= } is not an np.ndarray ')
            X_pred= self.dataset_handler.format_single_X(single_X, metadata)
            
        else:
            
            X_pred=self.dataset_handler.get_X('test')   
        return self.model_handler.predict(X_pred=X_pred, metadata=metadata, **kwargs)
        

    def save_model(self, metadata: MetaData) -> None:
        model_saving_path = self.model_handler.save(metadata=metadata)
        metadata.set_model_saving_path(model_saving_path)


    def load_model(self, metadata: MetaData) -> None:
        """select the model and dataset (need to be build after)"""

        self.select_model_dataset(metadata=metadata)
        self.model_handler.load(metadata=metadata)

if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)


    # X = np.random.randn(100, 4)
    # Y = np.random.randn(100)
    # Y = Y[:, np.newaxis]
    
    # rdn_dataset = PytorchDataset(X, Y)
    
    # test = StdPytorchDataset()
    
    # MetaData()
    
    # new_model = StdPytorchModelManager()
    # # for epoch in range(50):
    # new_model.train(test, 50)

    

    
