import pandas as pd
import numpy as np
from datetime import datetime as dt



def format_curr_pair(cur_pair):
    """
    Переделывает введенную пару в нужный формат
    """
    l = []
    l.append(cur_pair)
    l.append('_T+0')
    return ''.join(l)

def preparing_data(df):
    """
    Удаляет лишние столбцы
    """
    return df.drop(['exchange_time'], axis=1, inplace=True)

def place_cur(cur_pair, number):
    """
    Выдает валюту из пары
    number = 0 -- первая валюта в паре
    number = 1 -- вторая валюта в паре
    Пример использования:
    place_cur(currency_pairs_tob[1], 1)
    """
    if number == 0:
        start = 0
        end = cur_pair.find("/")
    if number == 1:
        start = cur_pair.find("/") + 1
        end = cur_pair.find("_T+0", start)
    return cur_pair[start:end]

def cross_section_data(df, currency_pair):
    return df.loc[df['instrument'] == currency_pair]
def loading_data(currency_pair):
    """
    Загрузка данных для данной валютной пары
    """
    # ограничитель на кол-во строк
    # nrows = 10**6
    nrows = int(input("Enter a limit on the number of rows, if -1, then there will be no restrictions:"))

    # 'Data/tob/tob_2022-09-21.csv'
    file_name_tob = input("Enter the file name for the tob-data:")
    file_name_trade = input("Enter the file name for the trade-data:")

    currency_pair = format_curr_pair(currency_pair)

    if nrows == -1:
        #tob
        df_tob = pd.read_csv(file_name_tob, delimiter=';')
        df_tob = preparing_data(df_tob)
        df_tob = df_tob.loc[df_tob['instrument'] == currency_pair]
        #trade
        df_trade = pd.read_csv(file_name_trade, delimiter=';')
        df_trade = preparing_data(df_trade)
        df_trade = df_trade.loc[df_trade['instrument'] == currency_pair]

    if nrows > 0:
        #tob
        df_tob = pd.read_csv(file_name_tob, delimiter=';', nrows=nrows)
        df_tob = preparing_data(df_tob)
        df_tob = df_tob.loc[df_tob['instrument'] == currency_pair]
        #trade
        df_trade = pd.read_csv(file_name_trade, delimiter=';', nrows=nrows)
        df_trade = preparing_data(df_trade)
        df_trade = df_trade.loc[df_trade['instrument'] == currency_pair]

    return df_tob, df_trade

