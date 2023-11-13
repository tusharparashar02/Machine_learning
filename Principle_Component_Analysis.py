import numpy as np
import matplotlib.pyplot as plt


data = np.array([[2,3],
                 [3,4],
                 [4,5],
                 [5,6],
                 [6,7],])


#normalize data
mean = np.mean(data, axis=0)
std_dev = np.std(data, axis=0)
data_std = (data - mean) / std_dev



#  compute covariance matrix
cov_mat = np.cov(data_std.T)
print(cov_mat)


# compute eigen values and eigen vectors
eig_val, eig_vec = np.linalg.eig(cov_mat)
print(eig_val)


#  select principal components
principal_components = eig_vec[:, np.argmax(eig_val)]

#  project data
projected_data = principal_components.reshape(-1, 1)

#  transform data
transformed_data = data_std.dot(projected_data)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Data (2D)')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(data[:, 0], data_std[:, 1])



# plot transformed data
plt.subplot(1, 2, 2)
plt.title('Transformed Data (1D)')
plt.xlabel('x')
plt.scatter(transformed_data, np.zeros_like(transformed_data))
plt.ylabel('')
plt.tight_layout()
plt.show()
