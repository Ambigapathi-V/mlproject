import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
                
    except Exception as e:
        raise CustomException(e, sys)
    
from sklearn.model_selection import GridSearchCV

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    model_report = {}
    
    for model_name, model in models.items():
        try:
            if params.get(model_name):
                grid_search = GridSearchCV(estimator=model, param_grid=params[model_name], scoring='r2', cv=5)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model
            
            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            model_report[model_name] = score
        except Exception as e:
            raise CustomException(e, sys)

    return model_report



