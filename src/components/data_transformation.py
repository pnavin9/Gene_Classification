import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

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

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            pass
        except:
            pass
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # Counting K-mer Sequence
            
            def Kmers_funct(seq, size=6):
                return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]
            train_df['words'] = train_df.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
            test_df['words'] = test_df.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
            
            train_df = train_df.drop('sequence', axis=1)
            test_df = test_df.drop('sequence', axis=1)
            
            train_texts = list(train_df['words'])
            test_texts = list(test_df['words'])
            
            for item in range(len(train_texts)):
                train_texts[item] = ' '.join(train_texts[item])
            for item in range(len(test_texts)):
                test_texts[item] = ' '.join(test_texts[item])
            
            cv = CountVectorizer(ngram_range=(4,4))
            
            train_texts = [t.encode('utf-8') for t in train_texts]
            test_texts = [t.encode('utf-8') for t in test_texts]
            X_train = cv.fit_transform(train_texts)
            X_test = cv.fit_transform(test_texts)
            
            train_target = train_df.iloc[:, 0].values
            test_target = test_df.iloc[:, 0].values
            train_arr = [X_train,np.array(train_target)]
            test_arr = [X_test,np.array(test_target)]

            logging.info(f"Save processing Object")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
