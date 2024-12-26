## Customer Lifetime Value End-to-End Project

conda create -p venv python==3.12 -y

conda activate venv/


### Python app.py:
http://127.0.0.1:5000/predictdata

### Azure Web App:
https://clvprediction-h5hhawayf6ffaccj.canadacentral-01.azurewebsites.net/predictdata 


### Approach for the Project
1. Data Ingestion :

- In Data Ingestion phase the data is first read as csv.
- Then the data is split into training and testing and saved as csv file.

2. Data Transformation :

- In this phase a ColumnTransformer Pipeline is created.
- for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
- for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is - scaled with Standard Scaler.
- Target Encoding for "Description" column as it had over 3000 values. 
- This preprocessor is saved as pickle file.

3. Model Training :

- Models trained:
    1. Linear Regression
    2. Lasso
    3. Ridge
    4. K-Neighbors Regressor
    5. Decision Tree Regressor
    6. Random Forest Regressor
    7. XGBRegressor
    8. CatBoosting Regressor
    9. AdaBoost Regressor

- In this phase base model is tested . 
- The best model found was Random Forest.
- After this hyperparameter tuning is performed, again the best model was Random Forest
- This model is saved as pickle file.

4. Prediction Pipeline :

- This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation :

- Flask app is created with User Interface to predict the gemstone prices inside a Web Application.
- http://127.0.0.1:5000/predictdata

6. Microsoft Azure Deployment:

- https://clvprediction-h5hhawayf6ffaccj.canadacentral-01.azurewebsites.net/predictdata 

### Exploratory Data Analysis Notebook 

- https://github.com/saurabhkamal/clvproject/blob/main/notebook/1_CLV_EDA.ipynb


