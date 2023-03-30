from pydantic import create_model, BaseModel
from typing import Literal
import joblib
import pandas as pd
import os
from dvc import api
from io import StringIO

from common.config_handler import Config
from common.log_handler import Logger

logger = Logger()
config = Config()

CATEGORIC_MODEL = list(config.get.model.features.categoric)

SEPARATOR_CATEGORIC = config.get.model.separator_categoric
REQUEST_FEATURES = vars(config.get.backend.request_features) 

# request features
request_features_param = { k : (eval(v), ...) for k,v in REQUEST_FEATURES.items() } 
StrokePredRequest = create_model('StrokePredRequest', **request_features_param)

class StrokePredResponse(BaseModel):
    prediction: float

class StrokePredModel():

    def __init__(self):
        logger.log.info("Retrieving and unpackage model.")
        # model = api.read('model_delay_flight.pkl', remote=os.environ.get('MODEL_TRACK_NAME'))
        # self.model = joblib.load(StringIO(model))
        self.model = joblib.load(os.environ.get("MODEL_PATH"))
        self.feature_names = self.model.feature_names_final

    def predict(self, st_request: StrokePredRequest):
        x = self.__transform_input(st_request)
        prediction = self.model.predict(x)[0]
        return prediction
        
    def __transform_input(self, st_request: StrokePredRequest):
        req = st_request.dict()

        # categoric treatment
        req_cat = { k:v for k,v in req.items() if k in CATEGORIC_MODEL }
        mask_features = [ SEPARATOR_CATEGORIC.join([k,v]) for k,v in req_cat.items() ]
        x = { k : ([1] if k in mask_features else [0]) for k in self.feature_names }

        # numeric treatment
        req_num = { k:[v] for k,v in req.items() if not k in CATEGORIC_MODEL }
        x.update(req_num)

        x = pd.DataFrame.from_dict(x)
        
        return x
