import pandas as pd
import pyodbc

server = 'localhost, port_no'
databse = 'test_db'
username = 'system_name'
password = 'password_name'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL server}; SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
print(cnxn)
cursor = cnxn.cursor()

#Create empty list
dfl = []

#Create empty dataframe
df_all_data = pd.DataFrame()

#Start chunking
allsql = "select * from table_name"
for chunk in pd.read_sql(allsql, con=cnxn, chunksize=100000):
    #Start appending data chunks from SQL result set into list
    dfl.append(chunk)
    #print(chunk)
#Start appending data from list to dataframe
df_all_data = pd.contact(dfl, ignore_index = True)

import pyarrow.feather as feather
feather.write_feather(df_all_data, "df_all_data_file.feather", compression="zstd", compression_level=None, chunksize=None, version=2)

df_all_data_update = feather.read_feather("df_all_data_file.feather")

df_all_data_update.head()


























