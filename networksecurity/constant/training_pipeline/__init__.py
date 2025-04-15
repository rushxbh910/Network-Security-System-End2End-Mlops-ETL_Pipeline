import os
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN="Result"
PIPELINE_NAME: str="networksecurity"
ARTIFACT_DIR: str="Artifacts"
FILE_NAME: str="phisingData.csv"

TRAIN_FILE_NAME: str="train.csv"
TEST_FILE_NAME: str="test.csv"

SCHEMA_FILE_PATH =os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR=os.path.join("data_schema","schema.yaml")
MODEL_FILE_NAME:str="model.pkl"


DATA_INGESTION_COLLECTION_NAME:str="NetworkData"
DATA_INGESTION_DATABASE_NAME:str="RUSHXBH910"
DATA_INGESTION_DIR_NAME:str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str="feature_store"
DATA_INGESTION_INGESTED_DIR: str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float=0.2

DATA_VALIDATION_DIR_NAME: str= "data_validation"
DATA_VALIDATION_VALID_DIR: str= "validated"
DATA_VALIDATION_INVALID_DIR: str= "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str= "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str= "report.yaml"

DATA_TRANSFORMATION_DIR_NAME: str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str="transformed data"
DATA_TRANSFORMATION_OBJECT_DIR: str="transformed_object"
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str="train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str="test.npy"
## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
PREPROCESSING_OBJECT_FILE_NAME: str="preprocessing.pkl"

MODEL_TRAINER_DIR_NAME: str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str="trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float=0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD:float=0.05
