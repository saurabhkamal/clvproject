import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                #"XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(task_type="CPU",verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [16,32]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[0.1],
                    'subsample':[0.6, 0.8],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [16, 32]
                },
                "Linear Regression":{},
                #"XGBRegressor":{
                 #   'learning_rate':[.1,.01,.05,.001],
                  #  'n_estimators': [8,16,32,64,128,256]
                #},
                "CatBoosting Regressor":{
                    'depth': [6],
                    'learning_rate': [0.1],
                    'iterations': [50]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[0.1],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [16, 32]
                }
                
            }

            # Evaluate models and log progress
            logging.info("Starting model evaluation")
            model_report = {}
            for model_name, model in models.items():
                logging.info(f"Evaluating model: {model_name}")
                print(f"Training {model_name}...")  # Print progress to console
                try:
                    score = evaluate_models(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        models={model_name: model},
                        param={model_name: params.get(model_name, {})},
                    )
                    model_report[model_name] = max(score.values())  # Store best score
                    logging.info(f"{model_name} evaluation complete with R2: {model_report[model_name]:.4f}")
                    print(f"{model_name} completed. R2 score: {model_report[model_name]:.4f}")
                except Exception as e:
                    logging.warning(f"Error training {model_name}: {e}")

            # Find the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            #model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
             #                                models=models, param=params)
            
            ## To get best model score from dict
            #best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            #best_model_name = list(model_report.keys())[
             #   list(model_report.values()).index(best_model_score)
            #]
            #best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model achieved an acceptable R2 score")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}")
            print(f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            # Evaluate the best model on test data
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"Best model test R2 score: {r2_square:.4f}")

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)