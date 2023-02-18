import csv
import pandas as pd
import numpy as np
from datetime import datetime as dt

# 'Data/tob/tob_2022-09-21.csv'
def loading_data(file_name, nrows):
    if nrows == -1:
        df = pd.read_csv(file_name, delimiter=';')
    else:
        df = pd.read_csv(file_name, delimiter=';', nrows=nrows)

    return df

def iterator(df_tob, df_trade):
    pass
