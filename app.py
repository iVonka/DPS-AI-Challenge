import pickle
import boto3
import os

from fastapi import FastAPI
from pydantic import BaseModel
#from dotenv import load_dotenv

#load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Create a Boto3 S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

bucket_name = os.getenv("S3_BUCKET")
file_key = os.getenv("MODEL_NAME")
local_file_path = "model.pkl"

# Download the model file from S3
s3.download_file(bucket_name, file_key, local_file_path)
print(f"Model downloaded to {local_file_path}")


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
