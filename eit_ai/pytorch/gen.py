
from logging import getLogger
from eit_ai.pytorch.dataset import PYTORCH_DATASETS
from eit_ai.raw_data.raw_samples import RawSamples
import numpy as np
from eit_ai.train_utils.gen import Generators, WrongDatasetError, WrongModelError, meas_duration
from eit_ai.pytorch.models import PyTorchModels, PYTORCH_MODELS

from eit_ai.train_utils.lists import ListGenerators, PytorchDatasets, PytorchModels
from eit_ai.train_utils.metadata import MetaData
# from eit_ai.train_utils.lists import KerasDatasets, KerasModels, ListModels, ListDatasets, ListGenerators



logger = getLogger(__name__)

################################################################################
# Keras Models
################################################################################

class GeneratorPyTorch(Generators):
    """ Generator class for keras models """
    def select_model_dataset(self, model_type: PytorchModels = None, dataset_type: PytorchDatasets = None,
                             metadata: MetaData = None):

        if model_type is None and dataset_type is None:
            model_type, dataset_type = metadata.model_type, metadata.dataset_type # get the data from metadata
        else:
            if isinstance(model_type, PytorchModels) and isinstance(model_type, PytorchDatasets):
                model_type, dataset_type = model_type.value, dataset_type.value# convert to string

        try:
            self.model_manager = PYTORCH_MODELS[PytorchModels(model_type)]()
        except ValueError:
            raise WrongModelError(f'Wrong model: {model_type}')

        try:
            self.dataset = PYTORCH_DATASETS[PytorchDatasets(dataset_type)]()
        except ValueError:
            raise WrongDatasetError(f'Wrong dataset: {dataset_type}')

        metadata.set_model_dataset_type(ListGenerators.Pytorch, PytorchModels(model_type), PytorchDatasets(dataset_type))


    def build_dataset(self, raw_samples: RawSamples, metadata: MetaData) -> None:
        """"""
        self.dataset.build(raw_samples, metadata)


    def build_model(self, metadata: MetaData) -> None:
        self.model_manager.build(metadata=metadata)


    def run_training(self, metadata: MetaData = None) -> None:
        _, duration = self._run_training(metadata, return_duration=True)
        metadata.set_training_duration(duration)
        logger.info(f'### Training lasted: {duration} ###')


    @meas_duration
    def _run_training(self, metadata: MetaData = None, **kwargs) -> None:
        self.model_manager.train(dataset=self.dataset, metadata=metadata)


    def get_prediction(self, metadata: MetaData, **kwargs) -> np.ndarray:
        prediction, duration = self._get_prediction(metadata, return_duration=True, **kwargs)
        # metadata.set_training_duration(duration)
        logger.info(f'### Prediction lasted: {duration} ###')
        return prediction


    @meas_duration
    def _get_prediction(self, metadata: MetaData, **kwargs) -> np.ndarray:
        return self.model_manager.predict(dataset=self.dataset, metadata=metadata, **kwargs)


    def save_model(self, metadata: MetaData) -> None:
        model_saving_path = self.model_manager.save(metadata=metadata)
        metadata.set_model_saving_path(model_saving_path)


    def load_model(self, metadata: MetaData) -> None:
        """select the model and dataset (need to be build after)"""

        self.select_model_dataset(metadata=metadata)
        self.model_manager.load(metadata=metadata)

if __name__ == "__main__":
    from glob_utils.log.log  import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)
    

    
