import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from Task2 import task2

# Load and normalize data
A, b = load_diabetes(return_X_y=True)
b = (b - np.mean(b)) / np.std(b)
A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)

# Store all errors across 10 runs
all_error_list_tr = []
all_error_list_te = []

# Run Task 2 ten times with different splits
for i in range(10):
    train_errors, test_errors = task2(A, b)
    all_error_list_tr.append(train_errors)
    all_error_list_te.append(test_errors)

    # Plot each run
    plt.figure(figsize=(8, 4))
    plt.plot(train_errors, label='Train Error')
    plt.plot(test_errors, label='Test Error')
    plt.title(f'Task 3 - Run {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

#Compute AVG and MIN curves 
min_num_of_iterations = min(len(l) for l in all_error_list_tr)

# Truncate all curves to the shortest length
all_error_list_tr = [a[:min_num_of_iterations] for a in all_error_list_tr]
all_error_list_te = [a[:min_num_of_iterations] for a in all_error_list_te]

#Convert to numpy arrays
arr_tr = np.array(all_error_list_tr)
arr_te = np.array(all_error_list_te)

#Average and minimum per iteration
avg_tr = np.average(arr_tr, axis=0)
avg_te = np.average(arr_te, axis=0)
min_tr = np.amin(arr_tr, axis=0)
min_te = np.amin(arr_te, axis=0)

#Plot AVG & MIN
plt.figure(figsize=(8, 5))
plt.plot(avg_tr, label='AVG train')
plt.plot(avg_te, label='AVG test')
plt.plot(min_tr, label='MIN train')
plt.plot(min_te, label='MIN test')
plt.title('Task 3 - AVG & MIN Errors')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()



