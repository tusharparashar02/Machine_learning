#warnings.filterwarnings('ignore')

# Import the numpy and pandas packag
import numpy as np
import pandas as pd


# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
advertising = pd.DataFrame(pd.read_csv("/content/Advertising.csv"))
advertising.head()
advertising.shape
advertising.info()
advertising.describe()
# Checking Null values
advertising.isnull().sum()*100/advertising.shape[0]
# There are no NULL values in the dataset, hence it is clean.
# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()
