import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        Description: str,
        Country: str,
        frequency: int,
        monetary_value: float,
        recency: int,
        Time: float):
        
        self.Description = Description

        self.Country = Country

        self.frequency = frequency

        self.monetary_value = monetary_value

        self.recency = recency

        self.Time = Time

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Description": [self.Description],
                "Country": [self.Country],
                "frequency": [self.frequency],
                "monetary_value": [self.monetary_value],
                "recency": [self.recency],
                "Time": [self.Time],
                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        