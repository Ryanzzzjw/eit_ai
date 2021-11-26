
import time
from datetime import timedelta
from logging import getLogger
from eit_ai.raw_data.raw_samples import RawSamples
import numpy as np
from eit_ai.train_utils.gen import Generators, WrongDatasetError, WrongModelError, meas_duration
from eit_ai.keras.models import KERAS_MODELS
from eit_ai.keras.dataset import KERAS_DATASETS
from eit_ai.train_utils.metadata import MetaData
from eit_ai.train_utils.lists import KerasDatasets, KerasModels, ListModels, ListDatasets, ListGenerators


logger = getLogger(__name__)

################################################################################
# Keras Models
################################################################################

class GeneratorKeras(Generators):
    """ Generator class for keras models """

    def select_model_dataset(self, model_type:KerasModels=None, dataset_type:KerasDatasets=None, metadata:MetaData=None):
        
        if model_type is None and dataset_type is None:
            model_type, dataset_type = metadata.model_type, metadata.dataset_type
        else:
            if isinstance(model_type, KerasModels) and isinstance(model_type, KerasDatasets):
                
                model_type, dataset_type = model_type.value, dataset_type.value

        try:
            self.model_manager=KERAS_MODELS[KerasModels(model_type)]()
        except ValueError:
            raise WrongModelError(f'Wrong model: {model_type}')

        try:
            self.dataset=KERAS_DATASETS[KerasDatasets(dataset_type)]()
        except ValueError:
            raise WrongDatasetError(f'Wrong dataset: {dataset_type}')

        metadata.set_model_dataset_type(ListGenerators.Keras, KerasModels(model_type), KerasDatasets(dataset_type))

    def build_dataset(self, raw_samples:RawSamples, metadata:MetaData)-> None:
        """"""
        self.dataset.build(raw_samples, metadata)

    def build_model(self, metadata:MetaData)-> None:
        self.model_manager.build(metadata=metadata)

    def run_training(self,metadata:MetaData=None)-> None:
        _, duration =self._run_training(metadata, return_duration=True)
        metadata.set_training_duration(duration)
        logger.info(f'### Training lasted: {duration} ###')

    @meas_duration
    def _run_training(self,metadata:MetaData=None,**kwargs)-> None:
        self.model_manager.train(dataset=self.dataset, metadata=metadata)

    def get_prediction(self,metadata:MetaData, **kwargs)-> np.ndarray:
        prediction, duration =self._get_prediction(metadata, return_duration=True, **kwargs)
        # metadata.set_training_duration(duration)
        logger.info(f'### Prediction lasted: {duration} ###')
        return prediction

    @meas_duration    
    def _get_prediction(self,metadata:MetaData, **kwargs)-> np.ndarray:
        return self.model_manager.predict(dataset=self.dataset, metadata=metadata,**kwargs)

    def save_model(self, metadata:MetaData)-> None:
        model_saving_path=self.model_manager.save(metadata=metadata)
        metadata.set_model_saving_path(model_saving_path)

    def load_model(self,metadata:MetaData)-> None:
        """select the model and dataset (need to be build after)"""

        self.select_model_dataset(metadata=metadata)
        self.model_manager.load(metadata=metadata)



if __name__ == "__main__":
    from eit_ai.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)
    """"""
    

    
