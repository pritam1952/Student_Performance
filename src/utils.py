import os
import sys 
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)   
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})
            if param_grid:  # if parameters exist, perform RandomizedSearchCV
                search = RandomizedSearchCV(model, param_distributions=param_grid, 
                                            n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            report[model_name] = {"model": best_model, "score": test_r2}

        return report

    except Exception as e:
        raise CustomException(e, sys)    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)     