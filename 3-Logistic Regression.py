import numpy
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
df = pd.read_csv('/content/50_Startups.csv')
x = df[['Marketing Spend']]
y = df['Profit']
logr = linear_model.LinearRegression()
logr.fit(x,y)
new_spends = [[254597.82],[402651.34]]
predicted_profit = logr.predict(new_spends)
print("Predicted profit according the marketing spend:")
for size, profit in zip (new_spends, predicted_profit):
print(f"profit: {size[0]}, Predicted profit: {profit:.2f}")
plt.scatter(x,y, color="blue", label='Actual Prices')
plt.scatter(600,1551.53, color="green", label='Predicted Price')
plt.plot(x,logr.predict (x), color='red', linewidth=2, label='Linear 
Regression')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.legend()
plt.title('Linear Regression: Marketing Spend vs Profit')
plt.show()

