import pandas as pd
import joblib

from pathlib import Path

model = joblib.load(Path("model_dir") / "model.joblib")

encode_Education_Level = joblib.load(
    Path("encoded_dir") / "encode_Education_Level.joblib"
)

encode_Industry = joblib.load(Path("encoded_dir") / "encode_Industry.joblib")

encode_Job_Role = joblib.load(Path("encoded_dir") / "encode_Job_Role.joblib")

encode_Company_Size = joblib.load(Path("encoded_dir") / "encode_Company_Size.joblib")

encode_Job_Level = joblib.load(Path("encoded_dir") / "encode_Job_Level.joblib")

encode_AI_Adoption_Level = joblib.load(
    Path("encoded_dir") / "encode_AI_Adoption_Level.joblib"
)

encode_Layoff_Risk = joblib.load(Path("encoded_dir") / "encode_Layoff_Risk.joblib")


def predict(input_data):
    df = pd.DataFrame([input_data])

    df["Education_Level"] = encode_Education_Level.transform(df["Education_Level"])
    df["Industry"] = encode_Industry.transform(df["Industry"])
    df["Job_Role"] = encode_Job_Role.transform(df["Job_Role"])
    df["Company_Size"] = encode_Company_Size.transform(df["Company_Size"])
    df["Job_Level"] = encode_Job_Level.transform(df["Job_Level"])
    df["AI_Adoption_Level"] = encode_AI_Adoption_Level.transform(
        df["AI_Adoption_Level"]
    )

    pred = model.predict(df)[0]

    prediction = encode_Layoff_Risk.inverse_transform([pred])[0]

    return {"prediction": prediction}


if __name__ == "__main__":
    # input_data = {
    #     "Age": 36,
    #     "Education_Level": "Bachelor's",
    #     "Years_of_Experience": 7,
    #     "Industry": "Retail",
    #     "Job_Role": "Store Manager",
    #     "Company_Size": "Medium",
    #     "Job_Level": "Senior",
    #     "Routine_Task_Percentage": 12,
    #     "Creativity_Requirement": 86,
    #     "Human_Interaction_Level": 71,
    #     "AI_Adoption_Level": "Low",
    #     "Number_of_AI_Tools_Used": 0,
    #     "AI_Usage_Hours_Per_Week": 1,
    #     "Tasks_Automated_Percentage": 6,
    #     "AI_Training_Hours": 3,
    # }
    # predict(input_data)
    pass
