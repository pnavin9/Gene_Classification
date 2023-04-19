from Bio import SeqIO
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from IPython import display
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from dataclasses import dataclass
import os
import sys
@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[0],
                train_array[-1],
                test_array[0],
                test_array[-1]
            )
            classifier = MultinomialNB(alpha=0.1)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            n = f1_score(y_test, y_pred,average = 'macro')
            if n < 0.6:
                raise CustomException("No best model found")
            print("F1 score =",n)
            save_object(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = classifier
            )
            return f1_score
        except Exception as e:
            raise CustomException(e,sys)
