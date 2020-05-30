#Import thu vien
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn import svm
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import joblib

#Nhap du lieu dau vao
print("import to file:")
filename=input()
result = pd.read_csv(filename)
data=pd.DataFrame(result)
#print(data.shape)

#Thong ke label
print(data.y.value_counts())

#Kiem tra du lieu thieu
print(data.isnull().sum())

#Chuan hoa data ve dang so de train model
data_new = pd.get_dummies(data, columns=['job','marital', 'education','default', 'housing','loan', 'contact','month', 'poutcome'])
data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)
#print(data_new.shape)

#Chia du lieu thanh test va train
y = pd.DataFrame(data_new['y'])
X = data_new.drop(['y'], axis=1)


#Train du lieu su dung Random Model va luu vao file save
rdclassifier = RandomForestClassifier()
rdclassifier.fit(X, y)
filename = 'save_rd.sav'
joblib.dump(rdclassifier, filename)
