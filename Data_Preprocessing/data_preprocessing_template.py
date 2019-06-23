# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:].values
y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer()
#use SimpleImputer instead
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoderx = LabelEncoder()
x[:,0] = labelencoderx.fit_transform(x[:,0])

# x[:,0,3] = labelencoderx.fit_transform(x[:,0,3])
ohe = OneHotEncoder(categorical_features=[0])
x = ohe.fit_transform(x).toarray()
x
