import fastapi
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import time
import json
import joblib

# OpenTelementry Imports
from opentelemetry import trace 
from opentelemetry.sdk.trace import TracerProvider 
from opentelemetry.sdk.trace.export import BatchSpanProcessor 
# from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter  # ✅ Still valid with opentelemetry-operations-python


__version__ = "1.0.0"
print(f"Starting Iris Prediction API version {__version__}")
# setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# setup structured logging
logger = logging.getLogger("demo-log-ml-cluster")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s",
}))

handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI app
app = FastAPI(title="Iris Prediction API",
              description="API for predicting Iris species based on flower measurements.",
              version="1.0.0",
              openapi_tags=[
                  {
                      "name": "probe",
                      "description": "Health checks for liveness and readiness."
                  },
                  {
                      "name": "prediction",
                      "description": "Endpoints for making predictions on Iris species."
                  }
              ])
print('Loading pre-trained model...')
# Load pre-trained model (assuming it's a joblib model)
# Replace 'iris_model.pkl' with the path to your actual model file  
model = joblib.load('model.joblib')  # Load your pre-trained model
print('Model loaded successfully!')

# Request model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
# Response model
class IrisResponse(BaseModel):
    species: str
    confidence: float
# Endpoint to handle Iris model predictions


def model_predict(input_data):
    # Simulate a model prediction
    import random
    species = random.choice(['setosa', 'versicolor', 'virginica'])
    confidence = round(random.uniform(0.5, 1.0), 2)
    return IrisResponse(species=species, confidence=confidence)


app_state = {'is_ready': False, 'is_alive': True}

@app.on_event('startup')
async def startup_event():
    import time 
    time.sleep(2)
    app_state['is_ready'] = True

@app.get('/live_check', tags=['probe'])
async def liveness_check():
    if app_state['is_alive']:
        return {'status': 'alive'}
    else:
        raise Response(status_code=503, content={'status': 'unhealthy'})
    

@app.get('/ready_check', tags=['probe'])
async def readiness_check():
    if app_state['is_ready']:
        return {'status': 'ready'}
    else:
        raise Response(status_code=503, content={'status': 'not ready'})
    
@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    # process_time = rond(time.time() - start_time
    duration = round(time.time() - start_time* 1000, 2)
    response.headers['X-Process-Time-ms'] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, '032x')
    logger.exception(json.dumps({
                                 
        "severity": "error",
        'event':'unhandled_exception',
        "message": str(exc),
        "trace_id": trace_id,
        'path': str(request.url),
        'err': str(exc),
        "request": {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": await request.body(),
        }
    }))
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "trace_id": trace_id,
        }
    )   


@app.api_route('/predict', methods=["GET", "POST"])
async def predict(request: Request, iris_request: IrisRequest):
    print('Received request for prediction:', iris_request)
    # Start a new span for the prediction operation
    with tracer.start_as_current_span("predict_iris"):
        start_time = time.time()
        trace_id = trace.get_current_span().get_span_context().trace_id

        try:
            input_data = iris_request.dict()
            result = model_predict(input_data)
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "severity": "info",
                "event": "prediction",
                "message": "Prediction successful",
                "trace_id": trace_id,
                "latency_ms": latency,
                "input_data": input_data,
                "result": result.dict()
            }))
            return result
        except ValidationError as e:
            logger.error(json.dumps({
                "severity": "error",
                "event": "validation_error",
                "message": "Validation error occurred",
                "trace_id": trace_id,
                "errors": e.errors(),
                "input_data": input_data
            }))
            raise HTTPException(status_code=422, detail=e.errors())
        except Exception as e:
            logger.exception(json.dumps({
                "severity": "error",
                "event": "prediction_error",
                "message": "Error during prediction",
                "trace_id": trace_id,
                "error": str(e),
                "input_data": input_data
            }))
            raise HTTPException(status_code=500, detail="Internal Server Error, Prediction failed")    

@app.get('/')
async def root():
    return {"message": "Welcome to the Iris Prediction API. Use /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200, log_level="info")
    # To run the app, use the command: uvicorn demo_log:app --reload
