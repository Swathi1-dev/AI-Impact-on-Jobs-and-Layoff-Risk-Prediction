import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import os
from dotenv import load_dotenv
import joblib
import logging
from pathlib import Path

load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_DIR"))
ENCODED_DIR = Path(os.getenv("ENCODED_DIR"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ENCODED_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(level=logging.INFO)


df = pd.read_csv(os.getenv("DATASET"))
logging.info("Data loaded")
# converting categorical columns into numerical using labelencoding

encode_Education_Level = LabelEncoder()
encode_Industry = LabelEncoder()
encode_Job_Role = LabelEncoder()
encode_Company_Size = LabelEncoder()
encode_Job_Level = LabelEncoder()
encode_AI_Adoption_Level = LabelEncoder()
encode_Layoff_Risk = LabelEncoder()

df["Education_Level"] = encode_Education_Level.fit_transform(df["Education_Level"])
df["Industry"] = encode_Industry.fit_transform(df["Industry"])
df["Job_Role"] = encode_Job_Role.fit_transform(df["Job_Role"])
df["Company_Size"] = encode_Company_Size.fit_transform(df["Company_Size"])
df["Job_Level"] = encode_Job_Level.fit_transform(df["Job_Level"])
df["AI_Adoption_Level"] = encode_AI_Adoption_Level.fit_transform(
    df["AI_Adoption_Level"]
)
df["Layoff_Risk"] = encode_Layoff_Risk.fit_transform(df["Layoff_Risk"])

logging.info("Data encoded")

# separate x and y

x = df.drop("Layoff_Risk", axis=1)
y = df.Layoff_Risk

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=float(os.getenv("TEST_SIZE")), random_state=42
)

model = XGBClassifier(
    booster="gbtree", eval_metric="auc", n_jobs=100, objective="binary:logistic"
)

model.fit(x_train, y_train)
logging.info("Model trained")

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
print("XGBOOST")
print("Accuracy score of training", accuracy_score(y_train, train_pred))
print("Accuracy score of testing", accuracy_score(y_test, test_pred))
print("\n")
sns.heatmap(confusion_matrix(y_test, test_pred), annot=True, fmt=".2f")
plt.xlabel("truth")
plt.ylabel("predict")

joblib.dump(model, MODEL_DIR / "model.joblib")

joblib.dump(encode_Education_Level, ENCODED_DIR / "encode_Education_Level.joblib")

joblib.dump(encode_Industry, ENCODED_DIR / "encode_Industry.joblib")

joblib.dump(encode_Job_Role, ENCODED_DIR / "encode_Job_Role.joblib")

joblib.dump(encode_Company_Size, ENCODED_DIR / "encode_Company_Size.joblib")

joblib.dump(encode_Job_Level, ENCODED_DIR / "encode_Job_Level.joblib")

joblib.dump(encode_AI_Adoption_Level, ENCODED_DIR / "encode_AI_Adoption_Level.joblib")

joblib.dump(encode_Layoff_Risk, ENCODED_DIR / "encode_Layoff_Risk.joblib")
logging.info("Models and encoded are saved repectively")
