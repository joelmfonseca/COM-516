import numpy as np
import scipy.io

# IMPORT YOUR OWN LIBRARIES TO RUN YOUR SIMULATED ANNEALING
from competition import mcmc
from linear_estimation import constant_schedule, get_reconstructed_error, relu


var = scipy.io.loadmat('observations.mat')
Y = var['Y']
Y = Y.reshape(Y.size)
W = var['W']
m = int(var['m'])
n = int(var['n'])

print(np.shape(Y))
print(np.shape(W))
print(m)
print(n)

# W_GENERATED = np.random.normal(size=(m, n))
# X_TRUE_GENERATED = np.random.choice(a=[-1, 1], size=(n, 1))
# Y_GENERATED = relu(np.matmul(W_matrix, X_vector_true) / np.sqrt(n))

# RUN YOUR SIMULATED ANNEALING
# x_hat IS THE ESTIMATE RETURNED BY YOUR ALGORITHM

BETA = 1
ITERATION_COUNT = 8000
for i in range(0, 8000, 100):
    beta_updated, energy, X_hat = mcmc(
                Y,
                W,
                n,
                1,
                constant_schedule,
                i
                )

    grand_truth = scipy.io.loadmat('ground_truth.mat')

    X_true = grand_truth['X']

    error = get_reconstructed_error(X_true, X_hat, n)
    print(error)
# scipy.io.savemat('YOUR_TEAM_NAME', {'x_estimate': X_hat}, appendmat=True, format='5', oned_as='column')
