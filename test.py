#Import thu vien
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA
import joblib




#Nhap du lieu dau vao
filename="test.csv"
result = pd.read_csv(filename)
data=pd.DataFrame(result)
#print(data.shape)

#Chuan hoa data ve dang so de train model
data_new = pd.get_dummies(data, columns=['job','marital', 'education','default', 'housing','loan', 'contact','month', 'poutcome'])
data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)
#print(data_new.shape)

#Chia du lieu thanh test va train
y = pd.DataFrame(data_new['y'])
X = data_new.drop(['y'], axis=1)

#Nhap du lieu train model
filesave="save.sav"
loaded_model = joblib.load(filesave)


print("Wait to load model")
#Chay model tren file test 
y_pred = loaded_model.predict(X)

#Nhap du lieu train model with PCA:
filesave2="save-pca.sav"
loaded_model2 = joblib.load(filesave2)

#Chuan hoa data theo PCA
pca_model = PCA(n_components=2) 
pca_model.fit(X)
X=pd.DataFrame(pca_model.transform(X))

#Chay model tren file test
y_pred2 = loaded_model2.predict(X)

# Accuracy va confuse martix
def testpoint():
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

def testpoint2():
    print(confusion_matrix(y, y_pred2))
    print(classification_report(y, y_pred2))

#Thong ke dataset tuong ung 3 truong hop( 1/ban dau, 2/sau chuan hoa va 3/sau chuan hoa PCA)
def thongke():
    print("Number of Instance:")
    print(len(data))
    print("Number of Attributes:")
    print(len(data.count()))
    print("Thong ke label:")
    print(data.y.value_counts())
    print("Lost data:")
    print(data.isnull().sum())

def thongke2():
    print("Number of Instance:")
    print(len(data_new))
    print("Number of Attributes:")
    print(len(data_new.count()))

def thongke3():
    print("Number of Instance:")
    print(len(X))
    print("Number of Attributes:")
    print(len(X.count()), "chua tinh label class")