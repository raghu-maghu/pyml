import category_encoders as ce
import pandas as pd

data = pd.DataFrame(data={'City':['Bomabay','Delhi','Bangalore','Mysore']})

print(data)

effect_coding = ce.sum_coding.SumEncoder(cols='City',verbose=True)

data_encoder = effect_coding.fit_transform(data)

print (data_encoder)