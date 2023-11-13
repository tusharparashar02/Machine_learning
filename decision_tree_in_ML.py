# decision tree in ml

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv")

data.quality.unique()

data.quality.value_counts()
print(data.quality.unique())

X = data.drop(columns=('quality'))
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(dt.score(X_test, y_test))


y_pred = dt.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print("confusion matrix: ", confusion_matrix(y_test, y_pred))

print("classification report: ", classification_report(y_test, y_pred))

from sklearn.tree import plot_tree

plt.figure(figsize=(40,40))
plot_tree(dt)
plt.show()
