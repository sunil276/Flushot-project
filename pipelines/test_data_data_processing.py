# import libraries
import pandas as pd
import os
import pickle
import logging

# Import Modules
import helper

# Import logger
logger = logging.getLogger("newfile.log")

def run():
    try:
        # Part - 1
        logger.info("Started loading test data")
        data = pd.read_csv(r"D:\Sunil\DrivenData\Flushotlearning-1\test_set_features.csv")
        trainings_labels=pd.read_csv(r"D:\Sunil\DrivenData\FluShotLearning\training_set_labels.csv")

        logger.info("Successfully loaded the data !!")
        logger.info("Started processing test data")
        numerical_cols = data.select_dtypes(include=['float64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        data = helper.replace_nan(data, numerical_cols, 9999)
        data = helper.replace_nan(data, categorical_cols, 'XYZ')
        columns_list = ['age_group',
                        'education', 'race', 'sex', 'income_poverty', 'marital_status',
                        'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry',
                        'employment_occupation']
        for i in columns_list:
            with open(fr".\dependency_files\label_encoder_{i}.pkl",'rb') as f:
                transformer = pickle.load(f)
                data[i]=transformer.transform(data[i])

        logger.info("Data processing on test data is successfully completed !!")
                

        # Part - 2
        final_data=pd.merge(data,trainings_labels,on="respondent_id",how="left")
        print(final_data)
        final_data.to_excel("Final Data/final_test_data.xlsx",index=False)
        logger.info("final test data is successfully saved !!")

        return final_data



    except Exception as e:
        print(e)
