from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import re

from common.log_handler import Logger

logger = Logger()

def data_transformer(func, *args, **kwargs):
    def inner(*args, **kwargs):
        ann_types = func.__annotations__.values()
        assert ann_types and all([ t==pd.DataFrame for t in ann_types ]), \
            f"Transformation must has all annotations with pd.DataFrame type."
        assert func.__code__.co_argcount == 1, f"Transformation must has just one parameter."
        if args : assert isinstance(args[0], pd.DataFrame), f"Transformation parameter must be pandas.DataFrame type."
        if kwargs : assert isinstance(list(kwargs.values())[0], pd.DataFrame), f"Transformation parameter must be pandas.DataFrame type."
        res = func(*args, **kwargs)
        assert isinstance(res, pd.DataFrame), f"Transformation must return pandas.DataFrame type."
        return res
    return inner

class DataTransformer(ABC):

    @data_transformer
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def __str__(self):
        return self.__class__.__name__

class DataPreparer():

    def __init__(self, data_transformers: List[DataTransformer] = []):
        self.data_transformers = data_transformers

    def add_data_transformer(self, data_transformer: DataTransformer):
        self.data_transformers.append(data_transformer)
        return self

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared_data = data.copy()
        for dt in self.data_transformers:
            logger.log.info(f'Applying transformation: {dt}')
            prepared_data = dt.transform(prepared_data)

        return prepared_data

class StringTransformer():

    @staticmethod
    def normalize_str(x):
        x = str(x)
        x = x.upper()
        x = re.split('[^a-zA-Z]', x)
        x = '_'.join([ c for c in x if c ])
        return x
