from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the model
model = pickle.load(open("model/model.pkl", "rb"))

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development. Replace "*" with specific domains in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define classes for input
class LoanApplication(BaseModel):
    person_age: float
    person_income: float
    loan_amnt: float
    loan_percent_income: float
    loan_intent: str
    person_home_ownership: str
    person_education: str

# Serve the UI
@app.get("/")
def serve_ui():
    """
    Serve the index.html file for the UI of our application.
    """
    return FileResponse("static/index.html")


# Endpoint for predictions
@app.post("/predict/")
def predict(application: LoanApplication):
    """
    Predict loan approval status based on input.
    """
    # Convert the input into a pandas DataFrame
    input_data = pd.DataFrame([application.dict()])

    # Generate predictions and probabilities
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Return the results as a JSON response
    return {
        "prediction": int(prediction[0]),
        "probability": probability[0].tolist()
    }
