import pickle
import uvicorn
from fastapi import FastAPI
from typing import Dict, Any
# to tell FastAPI that the input is a dictionary with string keys and any type of values, we use Dict[str, Any]
from typing import Literal
from pydantic import BaseModel, Field



class Customer(BaseModel):
    # Categorical features
    lead_source: Literal[
        'events',
        'organic_search',
        'paid_ads',
        'referral',
        'social_media'
    ]

    industry: Literal[
        'education',
        'finance',
        'healthcare',
        'manufacturing',
        'retail',
        'technology'
    ]

    employment_status: Literal[
        'employed',
        'self_employed',
        'student',
        'unemployed'
    ]

    location: Literal[
        'africa',
        'asia',
        'europe',
        'north_america',
        'south_america'
    ]

    # Numerical features
    annual_income: float = Field(ge=0, allow_inf_nan=False)
    number_of_courses_viewed: int = Field(ge=0)
    interaction_count: int = Field(ge=0)
    lead_score: float = Field(ge=0, allow_inf_nan=False)



class PredictResponse(BaseModel):
    converted_probability: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False
    )
    converted: bool


app = FastAPI(title="lead-scoring-prediction")

with open('model.bin' , 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer: Dict[str, Any]) -> float:
    result = pipeline.predict_proba([customer])[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer)-> PredictResponse:
    prob = predict_single(customer.model_dump())
    return PredictResponse(
        converted_probability= prob,
        converted= bool(prob >= 0.5)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)