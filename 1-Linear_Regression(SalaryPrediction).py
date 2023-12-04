import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
 'Age': [25, 30, 35, 40, 45, 50, 55, 60],
 'Years_of_Experience': [2, 5, 8, 10, 12, 15, 18, 20],
 'Salary': [50000, 60000, 75000, 80000, 90000, 100000, 110000, 120000]
}

# Convert data to numpy arrays
X = np.array(data['Age']).reshape(-1, 1)
X = np.array(data['Years_of_Experience']).reshape(-1,1)
y = np.array(data['Salary'])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)
new_employee_details = np.array([[30, 5]]).reshape(-1,1)
predicted_salary = model.predict(new_employee_details)

print("Predicted Salary:", predicted_salary[0])

# Visualize the data and the linear regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Linear Regression')
plt.scatter(new_employee_details[:, 0], predicted_salary, color='green', label='New Employee')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.title('Salary Prediction using Linear Regression')
plt.show()


# linear regression with random dataset

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


