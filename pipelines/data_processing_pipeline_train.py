# import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Import Modules
import helper

# Setting up the logger
import logging

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

def run():
    try:
        # Part - 1
        logger.info("Reading the files")
        data = pd.read_csv(r"C:\Users\Manu\Downloads\training_set_features.csv")
        trainings_labels=pd.read_csv(r"D:\Sunil\DrivenData\FluShotLearning\training_set_labels.csv")
        logger.info("Successfully loaded all the files !!")

        logger.info("Starting Data processing")

        numerical_cols = data.select_dtypes(include=['float64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        data = helper.replace_nan(data, numerical_cols, 9999)
        data = helper.replace_nan(data, categorical_cols, 'XYZ')
        logger.info("Data processing is completed !! ")
        columns_list = ['age_group',
                        'education', 'race', 'sex', 'income_poverty', 'marital_status',
                        'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry',
                        'employment_occupation']
        data=helper.label_encode(df=data, cols=columns_list)
        logger.info("All the pickle files have been saved in the Dependency folder successfully !!")

        # Part - 2
        final_data=pd.merge(data,trainings_labels,on="respondent_id",how="left")
        logger.info("Final training data has been successfully saved !!")
        return final_data


    except Exception as e:
        logger.info(e)
