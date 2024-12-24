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

    def get_data_transformer_object(self, train_df, target_column_name):
        '''
        Creates a ColumnTransformer for preprocessing the data.

        Parameters:
        target_encoding_map: dict
            A dictionary for target-guided encoding of the 'Description' column.
        '''
        try:
            numerical_columns = ['frequency', 'recency', 'Time', 'monetary_value']
            target_guided_column = 'Description'
            onehot_encoded_columns = ['Country']

            #

            # Dynamically compute target-guided encoding map
            target_encoding_map = (
                train_df.groupby(target_guided_column)[target_column_name]
                .mean()
                .to_dict()
            )
            logging.info(f"Target encoding map for '{target_guided_column}': {target_encoding_map}")

            # Define pipelines
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", PowerTransformer(method='yeo-johnson'))
            ])

            target_guided_pipeline = Pipeline([
                 ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                # Apply target-guided encoding by mapping values
                ("target_guided_encoder", 
                 SimpleImputer(strategy="constant", fill_value=0))  # Acts as a placeholder
            ])

            onehot_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Target-guided column: {target_guided_column}")
            logging.info(f"One-hot-encoded columns: {onehot_encoded_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("target_guide_pipeline", target_guided_pipeline, [target_guided_column]),
                ("onehot_pipeline", onehot_pipeline, onehot_encoded_columns)
            ])

            # Add the target-guided encoding map as an attribute to the pipeline
            preprocessor.target_encoding_map = target_encoding_map

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            target_column_name = 'CLV'
            preprocessing_obj = self.get_data_transformer_object(train_df, target_column_name)
            #numerical_columns = ['frequency', 'recency', 'Time', 'monetary_value']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes")

            # Validate dimensions before preprocessing
            logging.info(f"Shape of input features (train): {input_feature_train_df.shape}")
            logging.info(f"Shape of target features (train): {target_feature_train_df.shape}")

            # Apply preprocessing to train and test datasets
            input_feature_train_df['Description'] = input_feature_train_df['Description'].map(preprocessing_obj.target_encoding_map)
            input_feature_test_df['Description'] = input_feature_test_df['Description'].map(preprocessing_obj.target_encoding_map)

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
            #target_feature_train_array = np.array(target_feature_train_df).reshape(-1, 1)
            #logging.info(f"Shape of reshaped target features (train): {target_feature_train_array.shape}")
            
            
            # Concatenate processed features and target column for train data
            train_arr = np.c_[np.array(input_feature_train_arr), np.array(target_feature_train_df)]
            logging.info(f"Final training array shape: {train_arr.shape}")

            # Repeat for test data
            #target_feature_test_array = np.array(target_feature_test_df).reshape(-1, 1)
            
            test_arr = np.c_[np.array(input_feature_test_arr), np.array(target_feature_test_df)]
            
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
