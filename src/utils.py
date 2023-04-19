import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(X_train,y_train,X_test,y_test,classifier):
    try:
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        n = f1_score(y_test, y_pred,average = 'macro')  
        return n    
    except Exception as e:
        raise CustomException(e,sys)
