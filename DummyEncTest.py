import category_encoders as ce
import pandas as pd
data = pd.DataFrame(data={'City':['Bomabay','Delhi','Bangalore','Mysore']})
print(data)
dummy_enc= pd.get_dummies(data=data,drop_first=True)

print(dummy_enc)