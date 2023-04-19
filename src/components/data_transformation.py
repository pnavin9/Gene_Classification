import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
            df = pd.concat([train_df,test_df],ignore_index = True)
            def Kmers_funct(seq, size=6):
                return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]
            df['words'] = df.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
            #test_df['words'] = test_df.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
            
            df = df.drop('sequence', axis=1)
            #test_df = test_df.drop('sequence', axis=1)
            
            texts = list(df['words'])
            #test_texts = list(test_df['words'])
            
            for item in range(len(texts)):
                texts[item] = ' '.join(texts[item])
            #for item in range(len(test_texts)):
            #    test_texts[item] = ' '.join(test_texts[item])
            
            cv = CountVectorizer(ngram_range=(4,4))
            
            texts = [t.encode('utf-8') for t in texts]
            #test_texts = [t.encode('utf-8') for t in test_texts]
            X = cv.fit_transform(texts)
            #X_test = cv.fit_transform(test_texts)
            
            y_human = df.iloc[:, 0].values
            #test_target = test_df.iloc[:, 0].values
            X_train, X_test, train_target, test_target = train_test_split(X, 
                                                    y_human, 
                                                    test_size = 0.20, 
                                                    random_state=1)
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
