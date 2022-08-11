import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()


class Measurements(BaseModel):
    petal_length: float
    petal_width: float
    sepal_length: float
    sepal_width: float


with open("../models/iris_classification.model", "rb") as f:
    model = pickle.load(f)


@app.post('/')
async def scoring_endpoint(item: Measurements):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    prediction = model.predict(df)
    return dict(prediction=prediction[0])
