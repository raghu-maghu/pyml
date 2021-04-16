import pandas as pd
import numpy as np

data = pd.DataFrame({'value': [2,45, -23, 85, 28, 2, 35, -12]})
print(data)
data['log+1'] = (data['value']+1).transform(np.log)
print(data)
data['log'] = (data['value']-data['value'].min()+1) .transform(np.log)
print(data)