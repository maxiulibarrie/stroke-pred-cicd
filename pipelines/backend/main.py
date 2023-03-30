from fastapi import FastAPI
from pipelines.backend.models import StrokePredModel, StrokePredRequest, StrokePredResponse

from common.log_handler import Logger

logger = Logger()

logger.log.info("Loading model.")
stroke_pred_model = StrokePredModel()

logger.log.info("Getting service up.")
app = FastAPI()

@app.get('/predict-stroke', response_model = StrokePredResponse)
async def predict_stroke(st_request: StrokePredRequest):
    prediction = stroke_pred_model.predict(st_request)
    response = { 'prediction' : prediction }

    logger.log.info(f"Prediction for request: {st_request.dict()}")
    logger.log.info(f"Response: {response}")

    return response
