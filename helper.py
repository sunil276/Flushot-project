import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def fillna(df:pd.DataFrame,cols:list,value):
    """
    Definition: fill the na values on the specified columns\n

    Required arguments : dataframe, columns, value
    """
    if df is not None and cols and value:
        df[cols]=df[cols].fillna(value)
        return df
    else:
        raise ValueError("Please provide the required arguments")
    

def replace_nan(df:pd.DataFrame,cols,value):
    """
    Args: df,columns and value

    Return: replace the nan or blank values with value argument
    """
    df[cols]=df[cols].replace([np.nan],value)
    return df


def label_encode(df:pd.DataFrame,cols:list):
    """
    Args: df, columns

    Return : encode the categories with the number, and dump the encoder to the specified path
    """
    cwd_path=os.getcwd()
    dump_path=os.path.join(cwd_path,"dependency_files")
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    le=LabelEncoder()
    for col in cols:
        df[col]=le.fit_transform(df[col])
        filename=os.path.join(dump_path,f"label_encoder_{col}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(le, f)  # Correct usage
    return df





