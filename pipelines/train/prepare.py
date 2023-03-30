import pandas as pd
import numpy as np
from datetime import datetime
import os
from dvc import api
from io import StringIO

from pipelines.train.models import DataTransformer, DataPreparer, StringTransformer

from common.config_handler import Config
from common.log_handler import Logger

logger = Logger()
config = Config()

NUMERIC_MODEL = list(config.get.model.features.numeric)
BINARY_MODEL = list(config.get.model.features.binary)
CATEGORIC_MODEL = list(config.get.model.features.categoric)
TARGET = config.get.model.features.target

def prepare_train_data():
    logger.log.info("Retrieving data.")
    # data = api.read('dataset_SCL.csv', remote=os.environ.get('DATA_TRACK_NAME'))
    # data = pd.read_csv(StringIO(data))
    data = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

    data_preparer = DataPreparer([
        NormalizeColumns(),
        RemoveNA(),
        NormalizeCategory(),
        RemoveUnnecessaryCategories(),
        SelectFeatures()
    ])

    train_data = data_preparer.prepare_data(data)

    logger.log.info("Saving training data.")
    train_data.to_csv(os.environ.get('TRAIN_DATA'), index=False)

class NormalizeColumns(DataTransformer):
    def transform(self, data):
        data.columns = [ StringTransformer.normalize_str(d) for d in data.columns ]
        return data

class NormalizeCategory(DataTransformer):
    def transform(self, data):
        for feature in CATEGORIC_MODEL:
            data[feature] = data[feature].apply(StringTransformer.normalize_str)
        return data
    
class RemoveNA(DataTransformer):
    def transform(self, data):
        return data.dropna()

class RemoveUnnecessaryCategories(DataTransformer):
    def transform(self, data):
        # OTHER in gender
        data = data[ ~data['GENDER'].isin(['OTHER']) ]
        return data

class SelectFeatures(DataTransformer):
    def transform(self, data):
        return data[ NUMERIC_MODEL + BINARY_MODEL + CATEGORIC_MODEL + [TARGET] ]
    