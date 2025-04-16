
"""
Task2 -
spliting the data 80/20 to train/test solving LS for the train
and evaluating on the test
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Loading the dataset
A, b = load_diabetes(return_X_y=True)
b = (b - np.mean(b)) / np.std(b) # normalize b
A = (A - np.mean(A, axis=0)) / np.std(A, axis=0) #normalize A

# spliting the data int train and test
A_tr, A_te, b_tr, b_te = train_test_split(A, b, test_size=0.2)
m_tr = A_tr.shape[0]
m_te = A_te.shape[0]

x, delta, eps = np.zeros(A.shape[1]), 1e-4, 1e-3 
errors_tr = [] # train errors
errors_te = [] # test errors

for step in range(5000):
    # A.T @ A @ x - A.T @ b => A.T(A @ x - b)
    
    eq_tr = A_tr @ x - b_tr
    gradient = (2 / m_tr) * (A_tr.T @ eq_tr)
    train_error = np.linalg.norm(eq_tr) ** 2 / m_tr
    test_error = np.linalg.norm((A_te @ x - b_te)) ** 2 / m_te

    errors_tr.append(train_error)
    errors_te.append(test_error)

    if np.linalg.norm(gradient) <= delta:
        break

    x = x - eps * gradient

# plotting the graph for the MSE
plt.figure(figsize=(8, 5))
plt.plot(errors_tr, label="Train MSE", linestyle='-')
plt.plot(errors_te, label="Test MSE", linestyle='--')
plt.title("Task 2: Train vs Test MSE over Gradient Descent Steps")
plt.xlabel("Step - k")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()