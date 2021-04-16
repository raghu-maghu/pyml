import pandas as pd
import numpy as np

# to read the data in the csv file
data = pd.read_csv("C:/Users/rs6134/Downloads/data5.csv")
print(data, "n")

# making an array of all the attributes
d = np.array(data)[:, :-1]
print("n The attributes are: ", d)

# segragating the target that has positive and negative examples
target = np.array(data)[:, -1]
print("n The target is: ", target)

def candidateElimination(c,t):
    for i,val in enumerate(t):
        if val == "Yes":
            specific_hyp = c[i]
            break
    print("Specific hypothesis\n",specific_hyp)

    for i,val in enumerate(c):
        if t[i] == 'Yes':
            test_array = c[i]
            for j in range(len(specific_hyp)):
                if val[j] != specific_hyp[j]:
                    specific_hyp[j] = '?'
                else:
                    pass

    print(specific_hyp)

candidateElimination(d,target)


