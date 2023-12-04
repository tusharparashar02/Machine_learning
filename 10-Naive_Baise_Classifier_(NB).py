#Naive baise code in ml

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv")

data.quality.unique()

data.quality.value_counts()
print(data.quality.unique())

X = data.drop(columns=('quality'))
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
print(nb.score(X_test, y_test))



y_pred = nb.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report

print("confusion matrix: ", confusion_matrix(y_test, y_pred))

print("classification report: ", classification_report(y_test, y_pred))
