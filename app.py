import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the API
app = FastAPI()


# Define the input schema
class InputData(BaseModel):
    year: int
    month: int


@app.post('/predict')
async def predict(data: InputData):
    # Extract input data
    year = data.year
    month = data.month

    prediction = model.predict([[28, 23.08, -42.86, 35]])

    # Return the prediction
    return {'prediction': prediction[0]}
