from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the model from our model folder
model = pickle.load(open("model/model.pkl", "rb"))

# Initzlise fastAPI
app = FastAPI()

# Define classes for our input
class LoanApplication(BaseModel):
    person_age: float
    person_income: float
    loan_amnt: float
    loan_percent_income: float
    loan_intent: str
    person_home_ownership: str
    person_education: str

# Endpoint for the predictions
@app.post("/predict/")
def predict(application: LoanApplication):

    input_data = pd.DataFrame([application.dict()])


    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)


    return {
        "prediction": int(prediction[0]),
        "probability": probability[0].tolist()
    }

# Test endpoint
@app.get("/")
def root():
    return {"message": "Credit Scoring API is running!"}
