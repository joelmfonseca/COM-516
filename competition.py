import numpy as np
from tqdm import tqdm

from linear_estimation import get_energy, simulated_annealing


def competition_glauber_transition(Y_vector, X_vector, W_matrix, vector_size, beta):
    """Implementation of the Glauber or heat bath transition function without need of grand true"""

    index = np.random.choice(vector_size)
    energy_current = get_energy(Y_vector, X_vector, W_matrix, vector_size)
    deleted_value = X_vector[index]
    X_vector[index] = -X_vector[index]
    energy_next = get_energy(Y_vector, X_vector, W_matrix, vector_size)
    bernouilli_trial = np.random.binomial(n=1, p=((1+deleted_value*np.tanh(beta*(energy_next-energy_current)))/2))
    if not bernouilli_trial:
        X_vector[index] = 1
        energy_next = get_energy(Y_vector, X_vector, W_matrix, vector_size)
    else:
        X_vector[index] = -1
        energy_next = get_energy(Y_vector, X_vector, W_matrix, vector_size)
    return energy_next, X_vector


def mcmc(Y_vector, W_matrix, vector_size, beta, schedule_function, num_iter_mcmc):
    '''Apply the Markov Chain Monte-Carlo (MCMC) method.'''

    X_vector = np.random.choice(a=[-1,1], size=(vector_size,1))
    energy_acc = []
    beta_acc = []
    beta_iter = beta
    counter = 0
    for i in tqdm(range(num_iter_mcmc), desc='MCMC iterations [' + str(num_iter_mcmc) + ']'):
        if not counter:
            counter = 0
            # counter = np.random.choice(a=np.arange(N/2,3*N/2))
            beta_iter = simulated_annealing(beta, i, schedule_function)
        else:
            counter -= 1

        # apply the transition function
        energy, X_vector = competition_glauber_transition(Y_vector, X_vector, W_matrix, vector_size, beta_iter)
        beta_acc.append(beta_iter)
        energy_acc.append(energy)

    return beta_acc, energy_acc, X_vector
