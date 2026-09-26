import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
from scipy.stats import pearsonr
from log_code import setup_logging
logger = setup_logging("fs")
from sklearn.feature_selection import VarianceThreshold

def select_best_columns(X_train,X_test,y_train,y_test):
    try:
        logger.info(f"Before constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"Before constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        # constant Technique
        var_obj = VarianceThreshold(threshold=0.0)
        var_obj.fit(X_train)
        constant_columns = X_train.columns[~var_obj.get_support()]
        logger.info(f"Columns to remove in constant technique : {constant_columns}")

        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)
        logger.info(f"After constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"After constant Technique X_test columns and shape :{X_test.columns} : {X_test.shape}")

    # quasi quatum FEATURE SELECTION

        quasi_obj = VarianceThreshold(threshold=0.1)
        quasi_obj.fit(X_train)

        quasi_constant_columns = X_train.columns[~quasi_obj.get_support()]

        logger.info(f"Columns to remove in quasi constant technique : {quasi_constant_columns}")

        X_train = X_train.drop(columns=quasi_constant_columns)
        X_test = X_test.drop(columns=quasi_constant_columns)

        logger.info(f"After Quasi constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")

        logger.info(f"After Quasi constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

    #  CORRELATION / HYPOTHESIS TESTING

        corr_p_values = []

        for i in X_train.columns:
            values = pearsonr(X_train[i], y_train)
            corr_p_values.append(values)
        corr_p_values = np.array(corr_p_values)

        p_values = corr_p_values[:, 1]

        logger.info("p_values for each feature:")

        for column, p_value in zip(X_train.columns, p_values):
            logger.info(f"{column} : {p_value}")


    # REMOVE FEATURES BASED ON P-VALUE

        significance_level = 0.05
        columns_to_remove = X_train.columns[p_values > significance_level]

        logger.info(f"Columns removed based on hypothesis testing : {columns_to_remove}")

        X_train = X_train.drop(columns=columns_to_remove)
        X_test = X_test.drop(columns=columns_to_remove)

        logger.info(f"After Hypothesis Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")

        logger.info(f"After Hypothesis Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'error in line no: {er_line.tb_lineno} due to: {er_msg} reason : {er_type}')