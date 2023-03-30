import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import os

from common.config_handler import Config
from common.log_handler import Logger
from pipelines.train.reports import confusion_matrix_pretty, classification_report_pretty

logger = Logger()
config = Config()

TEST_SIZE = config.get.train.test_size_split
MODEL_TYPE = config.get.train.model_type
SEPARATOR_CATEGORIC = config.get.model.separator_categoric

PARAMETERS_GRID_SEARCH_SVM = vars(config.get.train.parameters_GridSearch.SVM)
PARAMETERS_GRID_SEARCH_XGBOOST = vars(config.get.train.parameters_GridSearch.XGBOOST)

NUMERIC_MODEL = list(config.get.model.features.numeric)
BINARY_MODEL = list(config.get.model.features.binary)
CATEGORIC_MODEL = list(config.get.model.features.categoric)
TARGET = config.get.model.features.target

MODEL_TYPE_LIST = config.get.train.model_type_list

def train():
    if not MODEL_TYPE in MODEL_TYPE_LIST:
        msg = f"Model Type: {MODEL_TYPE} does not exists or is not implemented."
        logger.log.error(msg)
        raise RuntimeError(msg)
    
    logger.log.info("Retrieving training data.")
    data = pd.read_csv(os.environ.get('TRAIN_DATA'))

    data = shuffle(data, random_state = 111)

    dummies = pd.concat([ 
        pd.get_dummies(data[col], prefix = col, prefix_sep = SEPARATOR_CATEGORIC) 
        for col in CATEGORIC_MODEL
    ], axis = 1) 

    features = pd.concat([
        dummies, data[BINARY_MODEL], data[NUMERIC_MODEL]
    ], axis = 1)

    label = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(features, label, 
                                                        test_size = TEST_SIZE, 
                                                        random_state = 42)

    model = None
    logger.log.info(f"Training {MODEL_TYPE} model.")
    if MODEL_TYPE == 'SVM':
        gs_svm_model = GridSearchCV(SVC(), \
            param_grid = PARAMETERS_GRID_SEARCH_SVM, \
            cv = 2, n_jobs=-1, verbose=1)

        gs_svm_model = gs_svm_model.fit(x_train, y_train)

        model = gs_svm_model.best_estimator_

    elif MODEL_TYPE == 'XGBOOST':
        gs_xgb_model = GridSearchCV(xgb.XGBClassifier(), \
            param_grid = PARAMETERS_GRID_SEARCH_XGBOOST, \
            cv = 2, n_jobs=-1, verbose=1)

        gs_xgb_model = gs_xgb_model.fit(x_train, y_train)

        model = gs_xgb_model.best_estimator_

    model.feature_names_final = list(x_train.columns.values)

    y_pred = model.predict(x_test)

    conf_m_report = confusion_matrix_pretty(y_test, y_pred)
    clf_report = classification_report_pretty(y_test, y_pred)

    logger.log.info("Confussion Matrix:")
    logger.log.info("\n" + str(conf_m_report))

    logger.log.info("Classification Report:")
    logger.log.info("\n" + str(clf_report))

    logger.log.info("Saving model.")
    joblib.dump(model, os.environ.get('MODEL_PATH'))
