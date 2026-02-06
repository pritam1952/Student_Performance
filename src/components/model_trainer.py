import os
import sys
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
# from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spiliting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                # "XGBRegressor": XGBRegressor(), 
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Linear Regression": {"fit_intercept": [True, False], "positive": [True, False]},
                "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10], "max_iter": [1000, 5000], "tol": [1e-4, 1e-3]},
                "Ridge": {"alpha": [0.1, 1, 10], "solver": ["auto", "svd", "lsqr"], "tol": [1e-4, 1e-3]},
                "K-Neighbors Regressor": {"n_neighbors": [3,5,7,9], "weights": ["uniform","distance"], "metric":["euclidean","manhattan"], "p":[1,2]},
                "Decision Tree": {"criterion": ["squared_error", "friedman_mse"], "max_depth": [None,5,10,20],
                                  "min_samples_split": [2,5,10], "min_samples_leaf": [1,2,4]},
                "Random Forest Regressor": {"n_estimators":[50,100,200], "criterion":["squared_error","friedman_mse"],
                                            "max_depth":[None,10,20], "min_samples_split":[2,5,10], "min_samples_leaf":[1,2,4]},
                # "XGB Regressor": {"n_estimators":[100,200], "learning_rate":[0.01,0.05,0.1], "max_depth":[3,5,7],
                #                   "subsample":[0.6,0.8,1.0], "colsample_bytree":[0.6,0.8,1.0]},
                "AdaBoost Regressor": {"n_estimators":[50,100,200], "learning_rate":[0.01,0.05,0.1], "loss":["linear","square"]}
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            # Get the best model name based on score
            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]

            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            prdicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,prdicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)       