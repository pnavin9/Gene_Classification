import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    #train_dog_path = os.path.join('artifacts',"dog.txt")
    #train_chimpanzee_path = os.path.join('artifacts',"chimpanzee.txt")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        logging.info("Entered Data ingestion method")
        try:
            df = pd.read_table('notebook/Data/human.txt')
            #dog_dna = pd.read_table('notebook/Data/dog.txt')
            #chimp_dna = pd.read_table('notebook/Data/chimpanzee.txt')
            logging.info('Read Dataset')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            #os.makedirs(os.path.dirname(self.ingestion_config.train_dog_path), exist_ok = True)
            #dog_dna.to_table(self.ingestion_config.raw_data_path, index = False, header = True)
            #os.makedirs(os.path.dirname(self.ingestion_config.train_chimpanzee_path), exist_ok = True)
            #chimp_dna.to_table(self.ingestion_config.raw_data_path, index = False, header = True)
            #def Kmers_funct(seq, size=6):
            #    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]
            #convert our training data sequences into short overlapping k-mers of length 6. 
            #Lets do that for each species of data we have using our Kmers_funct function.

            #human_dna['words'] = human_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
            #human_dna = human_dna.drop('sequence', axis=1)

            #chimp_dna['words'] = chimp_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
            #chimp_dna = chimp_dna.drop('sequence', axis=1)

            #dog_dna['words'] = dog_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
            #dog_dna = dog_dna.drop('sequence', axis=1)   
            #human_texts = list(human_dna['words'])
            #for item in range(len(human_texts)):
            #    human_texts[item] = ' '.join(human_texts[item])
            #chimp_texts = list(chimp_dna['words'])
            #for item in range(len(chimp_texts)):
            #    chimp_texts[item] = ' '.join(chimp_texts[item])
            #dog_texts = list(dog_dna['words'])
            #for item in range(len(dog_texts)):
            #    dog_texts[item] = ' '.join(dog_texts[item])    
            #y_human = human_dna.iloc[:, 0].values
            #y_chim = chimp_dna.iloc[:, 0].values 
            #y_dog = dog_dna.iloc[:, 0].values 
            #cv = CountVectorizer(ngram_range=(4,4))
            #X = cv.fit_transform(human_texts)
            #X_chimp = cv.transform(chimp_texts)
            #X_dog = cv.transform(dog_texts)
            #from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(df, test_size = 0.20,random_state=1)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)

            logging.info("ingestion of data is done")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
