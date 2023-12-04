# implement knn model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# load data
df = pd.read_csv('winequality-red.csv')
df.head()
# check for null values
# df.isnull().sum()
df = pd.get_dummies(df, columns=['quality'])
df.head()
# split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)
# train model
y_train = y_train.astype('int')
y_test = y_test.astype('int')
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# save model
pickle.dump(knn, open('model.pkl', 'wb'))
