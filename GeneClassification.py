import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('c:/Users/rs6134/Documents/personal/ML-1/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

train_data = pd.read_csv('C:\\Users\\rs6134\\Documents\\personal\\ML-1\\input\\data_set_ALL_AML_train.csv')
test_data = pd.read_csv('C:/Users/rs6134/Documents/personal/ML-1/input/data_set_ALL_AML_independent.csv')
labels = pd.read_csv('C:/Users/rs6134/Documents/personal/ML-1/input/actual.csv')

print(f'columns for train data: {train_data.columns.tolist()}')

train_data.shape

train_data

train_data.info()

labels['cancer'].unique()

call_values = train_data['call'].unique()
print(f'Values for call columns: {call_values}')

print(f'The test data shape: {test_data.shape}')

test_data

test_data['call'].unique()

# remove columns that contain Call data
train_columns = [col for col in train_data.columns if "call" not in col]
test_columns = [col for col in test_data.columns if "call" not in col]

train_X = train_data[train_columns]
test_X = test_data[test_columns]

# transpose the dataframe so that each row is a patient and each column is a gene
train_X = train_X.T
test_X = test_X.T
train_X.head()
train_X.columns

train_X.columns = train_X.iloc[1]
train_X.columns

# delete Gene Description and Gene Accession Number
train_X = train_X.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

# do the same for test data
test_X.columns = test_X.iloc[1]
test_X = test_X.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

print(f'Shape for train data: {train_X.shape}')
print(f'Shape for test data: {test_X.shape}')

label_encoder = LabelEncoder()
labels['CancerColumn'] = label_encoder.fit_transform(labels['cancer'])

cancer_types_before_encoding = labels['cancer'].unique()
print('\n--- Cancer types before label encoding: \n', cancer_types_before_encoding)

cancer_types_after_encoding = labels['CancerColumn'].unique()
print('\n--- Cancer types after label encoding: \n', cancer_types_after_encoding)

#drop the 'cancer' column
labels = labels.drop('cancer', axis = 1)

# data sets for training
train_X = train_X.reset_index(drop=True)
train_y = labels[labels.patient <= 38].reset_index(drop=True)

# data sets for testing
test_X = test_X.reset_index(drop=True)
test_y = labels[labels.patient > 38].reset_index(drop=True)


train_y = train_y.iloc[:,1]
train_y = list(train_y)
test_y = test_y.iloc[:,1]
test_y = list(test_y)

train_y

test_y
# convert from integer to float
train_X = train_X.astype(float, 64)
test_X = test_X.astype(float, 64)

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.fit_transform(test_X)

pca = PCA()
train_X_scaled_copy = train_X_scaled
pca.fit_transform(train_X_scaled_copy)

total = sum(pca.explained_variance_)
no_features = 0
current_variance = 0
while current_variance / total < 0.90:
    current_variance += pca.explained_variance_[no_features]
    no_features = no_features + 1

print(f'{no_features} features explain around 90% of the variance')

pca = PCA(n_components=no_features)
train_X_pca = pca.fit_transform(train_X)
test_X_pca = pca.transform(test_X)

var_exp = pca.explained_variance_ratio_.cumsum()
var_exp = var_exp * 100
plt.bar(range(no_features), var_exp);
#plt.title("PCA")
plt.title('Explained Variance Ratio Vs Number of Features (PCA)')
plt.show()