import os
import pathlib
import joblib, uvicorn, argparse
from fastapi import FastAPI, Request

try:
    from models import hepsiburada
except:
    from fastapi_hepsiburada_prediction.models import hepsiburada


# Read models saved during train phase
current_dir = pathlib.Path(__file__).parent.resolve()
dirname = os.path.join(current_dir, 'saved_models')
estimator_hepsiburada_loaded = joblib.load(os.path.join(dirname, "randomforest_with_hepsiburada.pkl"))



app = FastAPI()

def make_hepsiburada_prediction(model, request):
    # parse input from request
    memory= request["memory"]
    ram= request["ram"]
    screen_size= request["screen_size"]
    power= request["power"]
    front_camera= request["front_camera"]
    rc1= request["rc1"]
    rc3= request["rc3"]
    rc5= request["rc5"]
    rc7= request["rc7"]


    # Make an input vector
    hepsiburada = [[memory, ram, screen_size, power, front_camera, rc1, rc3, rc5, rc7]]

    # Predict
    prediction = model.predict(hepsiburada)

    return prediction[0]

# Hepsiburada Prediction endpoint
@app.post("/prediction/hepsiburada")
def predict_hepsiburada(request: hepsiburada):
    prediction = make_hepsiburada_prediction(estimator_hepsiburada_loaded, request.dict())
    return {"result":prediction}

# Get client info
@app.get("/client")
def client_info(request: Request):
    client_host = request.client.host
    client_port = request.client.port
    return {"client_host": client_host,
            "client_port": client_port}