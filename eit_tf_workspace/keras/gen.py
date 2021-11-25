
from logging import getLogger

import numpy as np
from eit_tf_workspace.keras.dataset import KERAS_DATASETS
from eit_tf_workspace.keras.models import KERAS_MODELS
from eit_tf_workspace.raw_data.raw_samples import RawSamples
from eit_tf_workspace.train_utils.dataset import Datasets
from eit_tf_workspace.train_utils.gen import (Generators, WrongDatasetError,
                                              WrongModelError,
                                              WrongSingleXError, meas_duration)
from eit_tf_workspace.train_utils.lists import (KerasDatasets, KerasModels,
                                                ListDatasets, ListGenerators,
                                                ListModels)
from eit_tf_workspace.train_utils.metadata import MetaData

logger = getLogger(__name__)

################################################################################
# Keras Models
################################################################################

class GeneratorKeras(Generators):
    """ Generator class for keras models """

    def select_model_dataset(self, model_type:KerasModels=None, dataset_type:KerasDatasets=None, metadata:MetaData=None)-> None:
        if model_type is None and dataset_type is None:
            model_type, dataset_type = metadata.model_type, metadata.dataset_type
        elif isinstance(model_type, KerasModels) and isinstance(model_type, KerasDatasets):
            model_type, dataset_type = model_type.value, dataset_type.value
        try:
            self.model_man=KERAS_MODELS[KerasModels(model_type)]()
        except ValueError:
            raise WrongModelError(f'Wrong model: {model_type}')
        try:
            self.dataset=KERAS_DATASETS[KerasDatasets(dataset_type)]()
        except ValueError:
            raise WrongDatasetError(f'Wrong dataset: {dataset_type}')

        metadata.set_model_dataset_type(ListGenerators.Keras, KerasModels(model_type), KerasDatasets(dataset_type))

    def build_dataset(self, raw_samples:RawSamples, metadata:MetaData)-> None:
        self.dataset.build(raw_samples, metadata)

    def build_model(self, metadata:MetaData)-> None:
        self.model_man.build(metadata=metadata)

    def run_training(self,metadata:MetaData=None, dataset:Datasets=None)-> None:
        logger.info('### Training started: ... ###')
        _, duration =self._run_training(metadata,dataset=dataset, return_duration=True)
        metadata.set_training_duration(duration)
        logger.info(f'### Training lasted: {duration} ###')

    @meas_duration
    def _run_training(self,metadata:MetaData=None,**kwargs)-> None:
        dataset_2_train=self.dataset
        if 'dataset' in kwargs:
            passed_dataset= kwargs.pop('dataset')
            if passed_dataset and isinstance(passed_dataset, type(self.dataset)):
                dataset_2_train=passed_dataset
        self.model_man.train(dataset=dataset_2_train, metadata=metadata)

    def get_prediction(
        self,
        metadata:MetaData,
        dataset:Datasets=None,
        single_X:np.ndarray= None,
        **kwargs)-> np.ndarray:

        logger.info('### Prediction started: ... ###')
        prediction, duration =self._get_prediction(metadata, dataset= dataset,single_X=single_X, return_duration=True, **kwargs)
        logger.info(f'### Prediction lasted: {duration} ###')
        return prediction

    @meas_duration    
    def _get_prediction(
        self,
        metadata:MetaData,
        dataset:Datasets=None,
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

    def save_model(self, metadata:MetaData)-> None:
        model_saving_path=self.model_man.save(metadata=metadata)
        metadata.set_model_saving_path(model_saving_path)

    def load_model(self,metadata:MetaData)-> None:
        self.select_model_dataset(metadata=metadata)
        self.model_man.load(metadata=metadata)



if __name__ == "__main__":
    import logging

    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    """"""
    

    
