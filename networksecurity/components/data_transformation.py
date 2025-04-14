import sys,os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    @staticmethod    
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    #KNN imputer
    def get_data_transformer_object(cls)->Pipeline:
        logging.info("entered get_data_transformer_object of transformation class")

        try:
            imputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS) #** key value pair consideration
            logging.info(f"initialized knnimputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            processor:Pipeline=Pipeline([("imputer",imputer)])
            return processor

        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        
        try:
            logging.info("starting data transformation!")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ##training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]
            target_feature_train_df=target_feature_train_df.replace(-1,0)

            ##testing dataframe
            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]
            target_feature_test_df=target_feature_test_df.replace(-1,0)

            #knn imputer
            preprocessor=self.get_data_transformer_object()
            transformed_input_train_features=preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_features=preprocessor.transform(input_feature_test_df)

            train_arr=np.c_[transformed_input_train_features,np.array(target_feature_train_df)]
            test_arr=np.c_[transformed_input_test_features,np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.data_transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.data_transformed_test_file_path,array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.data_transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.data_transformed_test_file_path
            )

            return data_transformation_artifact




        except Exception as e:
            raise NetworkSecurityException(e,sys)