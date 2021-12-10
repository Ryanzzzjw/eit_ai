
from logging import getLogger

import numpy as np
from eit_ai.pytorch.dataset import PYTORCH_DATASETS
from eit_ai.pytorch.models import PYTORCH_MODELS, StdPytorchModelManager
from eit_ai.raw_data.raw_samples import RawSamples
from eit_ai.train_utils.dataset import AiDataset
from eit_ai.train_utils.gen import (Generators, WrongDatasetError,
                                    WrongSingleXError, meas_duration)
from eit_ai.train_utils.lists import (ListGenerators, ListPytorchDatasets,
                                      ListPytorchModels, get_from_dict)
from eit_ai.train_utils.metadata import MetaData

logger = getLogger(__name__)

################################################################################
# Pytorch Models
################################################################################

class GeneratorPyTorch(Generators):
    """ Generator class for pytorch models """
    def select_model_dataset(self, model_type: ListPytorchModels = None, dataset_type: ListPytorchDatasets = None,
                             metadata: MetaData = None):


        if model_type is None and dataset_type is None:
            model_type = metadata.model_type
            dataset_type = metadata.dataset_type

        model_man,listmodobj = get_from_dict(
            model_type, PYTORCH_MODELS, ListPytorchModels, True)
        dataset,listdataobj= get_from_dict(
            dataset_type,PYTORCH_DATASETS, ListPytorchDatasets, True)
        self.model_man = model_man()
        self.dataset = dataset()
        metadata.set_model_dataset_type(
            ListGenerators.PyTorch, listmodobj, listdataobj)

    def build_dataset(self, raw_samples: RawSamples, metadata: MetaData) -> None:
        """"""
        self.dataset.build(raw_samples, metadata)


    def build_model(self, metadata: MetaData) -> None:
        self.model_man.build(metadata=metadata)


    def run_training(self, metadata: MetaData = None) -> None:
        _, duration = self._run_training(metadata, return_duration=True)
        metadata.set_training_duration(duration)
        logger.info(f'### Training lasted: {duration} ###')


    @meas_duration
    def _run_training(self, metadata: MetaData = None, **kwargs) -> None:
        self.model_man.train(dataset=self.dataset, metadata=metadata)


    def get_prediction(
        self,
        metadata:MetaData,
        dataset:AiDataset=None,
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
        dataset:AiDataset=None,
        single_X:np.ndarray= None,
        **kwargs)-> np.ndarray:

        X_pred=self.dataset.get_X('test')
        # another dataset can be here predicted (only test part)
        if dataset is not None:
            if not isinstance(dataset, type(self.dataset)): 
                raise WrongDatasetError(
                    f'{dataset= } and {self.dataset} dont have same type...')
            X_pred=dataset.get_X('test')
        # Single passed X can be here predicted, after been formated
        if single_X is not None:
            if not isinstance(single_X, np.ndarray):
                raise WrongSingleXError(f'{single_X= } is not an np.ndarray ')
            X_pred= self.dataset.format_single_X(single_X, metadata)

        return self.model_man.predict(X_pred=X_pred, metadata=metadata, **kwargs)


    def save_model(self, metadata: MetaData) -> None:
        model_saving_path = self.model_man.save(metadata=metadata)
        metadata.set_model_saving_path(model_saving_path)


    def load_model(self, metadata: MetaData) -> None:
        """select the model and dataset (need to be build after)"""

        self.select_model_dataset(metadata=metadata)
        self.model_man.load(metadata=metadata)

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

    

    
