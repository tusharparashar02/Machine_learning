import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
%matplotlib inline

#import scikitplot as skl
sns.set()
data=pd.read_csv("winequality-red.csv")

# data-pd.read_csv("https://raw.githubusercontent.com/aniruddhachoudhury Red-in-Qualty

data.head()

data.columns

data.quality.unique()

data.quality.value_counts()

data.info()

data.describe()

data.head()

x = data.drop(columns=('quality'))

y = data['quality']

x.head()

y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

x_test.head()
x_train.head()
y_train.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)
print(scaler.mean_)

x_train_tf = scaler.transform(x_train)
x_train_tf

from sklearn.svm import SVC
model = SVC()

model.fit(x_train_tf,y_train)
model.score(x_train_tf,y_train)
x_test_tf = scaler.transform(x_test)
x_test_tf
y_predict = model.predict(x_test_tf)
y_predict
y_test

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predict)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test,y_predict)
print('confusion Matrix:')
print(cm)

classification_rep = classification_report(y_test,y_predict)
print("Classification Report:")
print(classification_rep)
