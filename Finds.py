import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/rs6134/Downloads/MLDATA/Data.csv")

data = np.array(df)[:,:-1]
target = np.array(df)[:,-1]

print("Test features\n",data)
print("Target Features\n", target)

def finds(c,t):
    specific_hyp = c[0].copy()
    for i,val in enumerate(t):
        if(val[i] == 'Yes'):
            specific_hyp = c[i].copy()
        break

    for i,val in enumerate(c):
        if t[i] == 'Yes':
            for x in range (len(specific_hyp)):
                if val[x] != specific_hyp[x]:
                    specific_hyp[x] = '?'
                else:
                    pass

    return specific_hyp;

print('Specific Hypothesis is \n', finds(data,target))
