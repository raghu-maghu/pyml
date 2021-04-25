import pandas as pd

data = pd.read_csv('C:/Users/rs6134/Downloads/MLDATA/data.csv')

age = data.iloc [:,1].values
salary = data.iloc[:,2].values

from sklearn.preprocessing import Binarizer

x = age.reshape(1,-1)
y = salary.reshape(1,-1)

Binarizer1 = Binarizer(35)
Binarizer2 = Binarizer(61000)

print(Binarizer1.fit_transform(x))
print(Binarizer1.fit_transform(y))