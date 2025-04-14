# required for using sklearn (use next line)
# !pip install scikit-learn

"""
task 1 -
Calculating the error on each step of the gradient descent 
algo and plotting a graph error value as function of steps.
we have 
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Loading the dataset
A, b = load_diabetes(return_X_y=True)
b = (b - np.mean(b)) / np.std(b) # normalizing the vector b
m = A.shape[0]

x, delta, eps = np.zeros(A.shape[1]), 1e-4, 0.1 

errors = [] # errors arrays for plotting the errors graph

for step in range(4000):
    
    # A.T @ A @ x - A.T @ b => A.T(A @ x - b)
    
    eq = A @ x - b
    gradient = (2 / m) * (A.T @ eq) 
    error = np.linalg.norm(eq) ** 2 / m # calculating the 2nd norm for Ax - b
    errors.append(error)
    
    # checking the stopping condition (gradient <= delta)
    if np.linalg.norm(gradient) <= delta:
        break

    x = x - eps * gradient

# plotting the graph for the errors 
plt.plot(errors, marker='o', linestyle='-')
plt.title("Error $\\|Ax_k - b\\|^2$ over Gradient Descent Steps")
plt.xlabel("Step - k")
plt.ylabel("MSE")
plt.grid(True)

final_step = len(errors) - 1
final_error = errors[-1]
plt.plot(final_step, final_error, 'ro', label=f'Final Error: {final_error:.2f}')
plt.legend()


plt.show()
