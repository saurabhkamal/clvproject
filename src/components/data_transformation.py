import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, train_df):
        '''
        Creates a ColumnTransformer for preprocessing the data.

        Parameters:
        target_encoding_map: dict
            A dictionary for target-guided encoding of the 'Description' column.
        '''
        try:
            numerical_columns = ['frequency', 'recency', 'Time', 'monetary_value']
            target_encoded_columns = ['Description']
            onehot_encoded_columns = ['Country']

            # Dynamically generate target_encoding_map from training dataset
            target_encoding_map = {desc: idx for idx, desc in enumerate(train_df['Description'].unique())}

            # Define pipelines
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", PowerTransformer(method='yeo-johnson'))
            ])

            target_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("target_encoder", OneHotEncoder(handle_unknown='ignore'))
            ])

            onehot_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Target-encoded columns: {target_encoded_columns}")
            logging.info(f"One-hot-encoded columns: {onehot_encoded_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("target_pipeline", target_pipeline, target_encoded_columns),
                ("onehot_pipeline", onehot_pipeline, onehot_encoded_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(train_df)

            target_column_name = 'CLV'
            #numerical_columns = ['frequency', 'recency', 'Time', 'monetary_value']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes")

            # Validate dimensions before preprocessing
            logging.info(f"Shape of input features (train): {input_feature_train_df.shape}")
            logging.info(f"Shape of target features (train): {target_feature_train_df.shape}")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert sparse matrices to dense if needed
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Validate dimensions after preprocessing
            logging.info(f"Shape of preprocessed input features (train): {input_feature_train_arr.shape}")
            logging.info(f"Shape of target features (train): {np.array(target_feature_train_df).shape}")

            # Reshape target feature to 2D
            target_feature_train_array = np.array(target_feature_train_df).reshape(-1, 1)
            logging.info(f"Shape of reshaped target features (train): {target_feature_train_array.shape}")
            
            
            # Concatenate processed features and target column for train data
            train_arr = np.c_[np.array(input_feature_train_arr), np.array(target_feature_train_array)]
            logging.info(f"Final training array shape: {train_arr.shape}")

            # Repeat for test data
            target_feature_test_array = np.array(target_feature_test_df).reshape(-1, 1)
            test_arr = np.c_[np.array(input_feature_test_arr), np.array(target_feature_test_array)]
            logging.info(f"Final testing array shape: {test_arr.shape}")
            
            logging.info(f"Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
