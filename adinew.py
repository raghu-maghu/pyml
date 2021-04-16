print("Hello  My Universe")
import pandas as pd
#C:/Users/rs6134/Downloads/data5.csv
dfx=pd.read_csv('C:/Users/rs6134/Downloads/MLDATA/data.csv')
print(dfx.head(4))
print(dfx.sum(axis = 0, skipna = True))