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

#Nhap du lieu train model Random Forest
filesave="save_rd.sav"
loaded_model = joblib.load(filesave)


print("Wait to load model")
#Chay model tren file test 
y_pred = loaded_model.predict(X)

#Nhap du lieu train model with SVM-PCA:
filesave2="save-svm.sav"
loaded_model2 = joblib.load(filesave2)


#Nhap du lieu train model with Logistic Regression-PCA
filesave3="save-lg.sav"
loaded_model3 = joblib.load(filesave3)

#Chuan hoa data theo PCA
pca_model = PCA(n_components=2) 
pca_model.fit(X)
X=pd.DataFrame(pca_model.transform(X))

#Chay model SVM-PCA tren file test 
y_pred2 = loaded_model2.predict(X)

#Chay model Logistic Regression-PCA tren file test
y_pred3 = loaded_model3.predict(X)

# Accuracy va confuse martix
def testpoint():
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

def testpoint2():
    print(confusion_matrix(y, y_pred2))
    print(classification_report(y, y_pred2))

def testpoint3():
    print(confusion_matrix(y, y_pred3))
    print(classification_report(y, y_pred3))

#Thong ke dataset tuong ung 3 truong hop( 1/ban dau, 2/sau chuan hoa RandomForest va 3/sau chuan hoa PCA(bao gom SVM va Logistic Regression))
def thongke(): #ban dau
    print("Number of Instance:")
    print(len(data))
    print("Number of Attributes:")
    print(len(data.count()))
    print("Thong ke label:")
    print(data.y.value_counts())
    print("Lost data:")
    print(data.isnull().sum())

def thongke2(): #sau chuan hoa dung cho RandomForest
    print("Number of Instance:")
    print(len(data_new))
    print("Number of Attributes:")
    print(len(data_new.count()))

def thongke3(): #sau chuan hoa PCA dung cho SVM va Logistic Regression
    print("Number of Instance:")
    print(len(X))
    print("Number of Attributes:")
    print(len(X.count()), "chua tinh label class")