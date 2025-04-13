from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config=DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e,sys)


    @staticmethod    
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        


    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Dataframe has columns:{len(dataframe.columns)}")

            if(number_of_columns==len(dataframe.columns)):
                return True
            else:
                return False

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_numerical_columns(self, dataframe)->bool:
        try:
            # Select all numerical columns
            numerical_columns = dataframe.select_dtypes(include=['number']).columns

            # Return True only if all columns are numerical
            return len(numerical_columns) == len(dataframe.columns)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}

            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})
            drift_report_file_path=self.data_validation_config.drift_report_file_path

            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

        except Exception as e:
            raise NetworkSecurityException(e,sys)



    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            ##read the the data from train.csv and test.csv
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            ##validate number of columns
            status=self.validate_number_of_columns(train_dataframe)
            if status!=True:
                error_message=f"Train dataframe does not contain all columns!\n"

            status=self.validate_number_of_columns(test_dataframe)
            if status!=True:
                error_message=f"Test dataframe does not contain all columns!\n"

            #check if all the columns are numerical
            status=self.validate_numerical_columns(train_dataframe)
            if status!=True:
                error_message=f"Not all the columns in the Train dataframe are numerical!\n"

            status=self.validate_numerical_columns(test_dataframe)
            if status!=True:
                error_message=f"Not all the columns in the Test dataframe are numerical!\n"

            #check data drift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact=DataValidationArtifact(
                validation_status=status,
                valid_test_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_train_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)