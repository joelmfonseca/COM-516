import numpy as np
import scipy.io

# IMPORT YOUR OWN LIBRARIES TO RUN YOUR SIMULATED ANNEALING
from linear_estimation import *
import time

var = scipy.io.loadmat('observations.mat')
Y = var['Y']
# print(Y.shape)
# Y = Y.reshape(Y.size)
# print(Y.shape)
W = var['W']
m = int(var['m'])
n = int(var['n'])

print(type(Y[0]), type(W))

print('Y.shape=',np.shape(Y))
print('W.shape=',np.shape(W))
print('M=',m)
print('N=',n)

alpha = m/n
print('alpha=', alpha)
beta = min(0.5, 1/alpha)
print('beta= {:.2f} ({:.2f})'.format(beta, 1/alpha))

# RUN YOUR SIMULATED ANNEALING
# X_hat IS THE ESTIMATE RETURNED BY YOUR ALGORITHM

def mcmc(Y_vector, W_matrix, vector_size, beta, transition_function, schedule_function, filename):
    '''Apply the Markov Chain Monte-Carlo (MCMC) method.'''

    X_vector = np.random.choice(a=[-1,1], size=(vector_size,1))
    energy_acc = []
    beta_iter = beta
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    energy = vector_size
    i=0
    start = time.time()
    while energy != 0:
        
        # update inverse temperature
        beta_iter = simulated_annealing(beta, i, schedule_function)

        # apply the transition function
        energy, X_hat = transition_function(Y_vector, X_vector, X_vector, W_matrix, vector_size, beta_iter)
        energy_acc.append(energy)
        i += 1

        if i%1000 == 0:
            end = time.time()
            elapsed_time = end-start
            print('iteration={},  energy={:.4f},  elpased time={:1d} min {:1d} sec'.format(i, energy/(alpha*vector_size), int(elapsed_time//60), int(elapsed_time%60)))
            plt.plot([e/(alpha*vector_size) for e in energy_acc], label='energy')
            plt.tight_layout()
            plt.savefig(filename + '.pdf')
            scipy.io.savemat('STATIONARY', {'x_estimate':X_hat}, appendmat=True, format='5', oned_as='column')
        


    end = time.time()
    elapsed_time = end-start

    print('Found X_true at iteration {} in {:1d} min {:1d} sec.'.format(i, int(elapsed_time//60), int(elapsed_time%60)))

    return X_hat

X_hat = mcmc(Y, W, n, beta, glauber_transition, linear_schedule, 'g_l_2')

scipy.io.savemat('STATIONARY', {'x_estimate':X_hat}, appendmat=True, format='5', oned_as='column')

# # check if found it:
# X_true = scipy.io.loadmat('ground_truth.mat')
# X_hat = scipy.io.loadmat('STATIONARY.mat')

# print(np.array_equal(X_true['X'], X_hat['x_estimate']))
# # print(list(zip([a[0] for a in X_true['X']], [b[0] for b in X_hat['x_estimate']])))