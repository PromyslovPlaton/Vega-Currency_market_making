import Data_instruments

# ограничитель на кол-во строк
#nrows = 10**6
nrows = int(input("Enter a limit on the number of rows, if -1, then there will be no restrictions:"))

# 'Data/tob/tob_2022-09-21.csv'
file_name_tob = input("Enter the file name for the tob-data:")
file_name_trade = input("Enter the file name for the trade-data:")

df_tob = Data_and_Iterator.loading_data(file_name_tob, nrows)
df_trade = Data_and_Iterator.loading_data(file_name_trade, nrows)



