'''
 i am going to developing ml classification project on
 heart diseace data set
'''
import os
import sys
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pandas.io import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging("main")
from yeo_timing import variable_transformation_outliers
from fs import select_best_columns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from all_models import common
from all_models import auc_roc_curve_structure
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle


class HEART_DISEASE_PROJECT:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f'total number of rows and columns : {self.df.shape}')
            logger.info(f'null values in the data : {self.df.isnull().sum()}')
            self.X = self.df.iloc[: , :-1]
            self.y = self.df.iloc[: , -1]
            logger.info(self.y.unique())
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)
            logger.info(f" training data size : \n {self.X_train.shape} =>  {self.y_train.shape}")
            logger.info(f" testing data size : \n {self.X_test.shape} => {self.y_test.shape}")



        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'error in line no: {er_line.tb_lineno} due to: {er_msg} reason : {er_type}')

    def vt_outliers(self):
        try:
            self.X_train, self.X_test = variable_transformation_outliers(self.X_train,self.X_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def features_selection(self):
        try:
            self.X_train, self.X_test = select_best_columns(self.X_train, self.X_test, self.y_train,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def data_balancing(self):
        try:
            logger.info(f"Total Final Training data shape : {self.y_train.shape}")
            logger.info(f"Number of Rows for Good :{1} : Class : {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for Bad :{0} : Class : {sum(self.y_train == 0)}")
            sm_obj = SMOTE(random_state=42)
            self.X_train_bal, self.y_train_bal = sm_obj.fit_resample(self.X_train, self.y_train)
            logger.info(f"After Balancing Training data shape : {self.y_train_bal.shape}")
            logger.info(f"Number of Rows for Good :{1} : Class : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for Bad :{0} : Class : {sum(self.y_train_bal == 0)}")
            # lets scale down the values
            sc = StandardScaler()
            sc.fit(self.X_train_bal)
            self.X_train_bal_scaled = sc.transform(self.X_train_bal)
            self.X_test_scaled = sc.transform(self.X_test)

            with open("scaled_model.pkl", "wb") as k:
                pickle.dump(sc, k)

            # know we can give the indepdent data and dependent data to ML Algorithms
            # Training_variables : (self.X_train_bal_scaled , self.y_train_bal)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def train_all_models(self):
        try:
            common( self.X_train_bal_scaled, self.y_train_bal, self.X_test_scaled, self.y_test)
            # since from the above line we trained all models and we got to know
            # Naive Bayes is working fine
            reg = GaussianNB()
            reg.fit(self.X_train_bal_scaled, self.y_train_bal)
            logger.info(f"the data accuracy by the model was: {accuracy_score(self.y_test, reg.predict(self.X_test_scaled))}")
            logger.info(f"the data confusion matrix by the model was: {confusion_matrix(self.y_test, reg.predict(self.X_test_scaled))}")
            logger.info(f"the data classification report by the model was: {classification_report(self.y_test, reg.predict(self.X_test_scaled))}")
            #save the model into pickle file
            with open("Heart_disease_model.pkl", "wb") as k1:
                pickle.dump(reg,  k1)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")





if __name__ == "__main__" :
    try:
        obj = HEART_DISEASE_PROJECT("heart.csv")
        obj.vt_outliers()
        obj.features_selection()
        obj.data_balancing()
        obj.train_all_models()

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'error in line no: {er_line.tb_lineno} due to: {er_msg} reason : {er_type}')



