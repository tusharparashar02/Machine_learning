import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#provided data
data={
    'Plot_size': [100,150,200,250,300,350,400,450,500],
    'Plot_price': [200000,250000,300000,350000,400000,450000,500000,550000,600000]
}
df=pd.DataFrame(data)
print(df)
#convert data to numpy array
x=df[['Plot_size']]
y=df['Plot_price']
# Create and fit the linear regression model
model=LinearRegression()
model.fit(x,y)=-
new_sizes=[[600], [700]]
predicted_prices=model.predict(new_sizes)
print('predicted prices :')
for size, price in zip(new_sizes, predicted_prices):
  print(f"Plot Size: {size[0]}, Predicted price: {price:.2f}")
  #visualize the data
plt.scatter(x,y, color='blue', label='Actual Prices')
plt.plot(x,model.predict(x), color='red', linewidth=2,label='Linear Regression')
plt.xlabel('Plot Size')
plt.ylabel('Plot Price')
plt.show()


