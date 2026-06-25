from fastapi import FastAPI
from pydantic import BaseModel
from src.prediction import predict

app = FastAPI()


class DataCheck(BaseModel):
    Age: int
    Education_Level: str
    Years_of_Experience: int
    Industry: str
    Job_Role: str
    Company_Size: str
    Job_Level: str
    Routine_Task_Percentage: int
    Creativity_Requirement: int
    Human_Interaction_Level: int
    AI_Adoption_Level: str
    Number_of_AI_Tools_Used: int
    AI_Usage_Hours_Per_Week: int
    Tasks_Automated_Percentage: int
    AI_Training_Hours: int


@app.get("/health")
def home():
    return {"success": "ok"}


@app.post("/predict-risk")
def prediction(data: DataCheck):

    result = predict(data.model_dump())

    return result
