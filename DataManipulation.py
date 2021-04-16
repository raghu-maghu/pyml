import pandas as pd
import numpy as np

dataframe = pd.read_csv("C:/Users/rs6134/Downloads/data5.csv");
print("Original Data \n", dataframe)
target_value = dataframe.iloc[:, -1]
source_value = dataframe.iloc[:, :-1]

print("Input vector \n", source_value)
print("Target Value \n",target_value)

def finds(c,t):
    for i,val in enumerate(t):
        if val == "Yes":
            specific_record = c[i]
        break
    print(specific_record)

finds(source_value,target_value)