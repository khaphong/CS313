#Import thu vien
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

#Nhap du lieu dau vao
print("import to file:")
filename="train.csv"
result = pd.read_csv(filename)
data=pd.DataFrame(result)
print(data.y.value_counts())

#Chuan hoa du lieu ve dang so de chay PCA
data_new = pd.get_dummies(data, columns=['job','marital', 'education','default', 'housing','loan', 'contact','month', 'poutcome'])
data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)
#print(data_new.shape)

#Tach label va thuc hien chay PCA lua chon n_components
y = pd.DataFrame(data_new['y'])
X = data_new.drop(['y'], axis=1)
pca_model = PCA(n_components=None) #n_components=None giu nguyen so chieu ko giam
pca_model.fit(X)
variance = pca_model.explained_variance_ratio_ #calculate variance ratios
var=np.cumsum(np.round(pca_model.explained_variance_ratio_, decimals=3)*100)

#Bieu do hien thi
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')


plt.plot(var)
plt.show()

print("Dua vao bieu do ta thay dc n_components=2 la tot nhat")
