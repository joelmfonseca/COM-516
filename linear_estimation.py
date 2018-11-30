# mini-project for MCAA course

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def relu(vector):
    '''Return the Rectifier Linear Unit (ReLU) function applied to a vector. It is defined as x-> max{0,x}.'''
    vector[vector < 0] = 0
    return vector

def get_energy(Y_vector, vector, W_matrix, vector_dim):
    '''Return the energy of a vector compared to an observation vector.'''
    return np.sum(np.abs(Y_vector - relu(np.matmul(W_matrix, vector) / np.sqrt(vector_dim))))

def get_reconstructed_error(X_vector_true, X_vector, vector_size):
    return np.linalg.norm(X_vector - X_vector_true)**2/(4*vector_size)

def gibbs_boltzmann_prob(delta_energy, beta):
    '''Return the Gibbs-Boltzmann probability.'''
    return np.exp(-beta*delta_energy)

def random_transition(Y_vector, X_vector_true, X_vector, W_matrix, vector_size, beta=0):
    '''Implementation of the random transition function.'''
    X_vector_random = np.random.choice(a=[-1,1], size=(vector_size,1))
    energy_next = get_energy(Y_vector, X_vector_random, W_matrix, vector_size)
    return energy_next, get_reconstructed_error(X_vector_true, X_vector_random, vector_size), X_vector_random

def metropolis_transition(Y_vector, X_vector_true, X_vector, W_matrix, vector_size, beta):
    '''Implementation of the Metropolis transition function.'''
    index = np.random.choice(vector_size)
    energy_current = get_energy(Y_vector, X_vector, W_matrix, vector_size)
    X_vector[index] = -X_vector[index]
    energy_next = get_energy(Y_vector, X_vector, W_matrix, vector_size)
    bernouilli_trial = np.random.binomial(n=1, p=np.min([1, gibbs_boltzmann_prob(energy_next-energy_current, beta)]))
    if not bernouilli_trial:
        X_vector[index] = -X_vector[index]
        energy_next = energy_current
    return energy_next, get_reconstructed_error(X_vector_true, X_vector, vector_size), X_vector

def glauber_transition(Y_vector, X_vector_true, X_vector, W_matrix, vector_size, beta):
    '''Implementation of the Glauber or heat bath transition function.'''
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
    return energy_next, get_reconstructed_error(X_vector_true, X_vector, vector_size), X_vector

def constant_schedule(beta, t):
    '''Implementation of a constant schedule for simulated annealing.'''
    return beta

def linear_schedule(beta, t):
    '''Implementation of a linear schedule for simulated annealing.'''
    step = 4/8000
    return beta + step*t

def exponential_schedule(beta, t):
    '''Implementation of an exponential schedule for simulated annealing.'''
    e = np.exp(np.log(5)/8000)
    return beta * e**t

def simulated_annealing(beta, t, schedule_function):
    '''Implementation of the simulated annealing method.'''
    return schedule_function(beta, t)

def mcmc(Y_vector, X_vector_true, W_matrix, vector_size, beta, transition_function, schedule_function, random_wait, num_iter_mcmc):
    '''Apply the Markov Chain Monte-Carlo (MCMC) method.'''
    X_vector = np.random.choice(a=[-1,1], size=(vector_size,1))
    energy_acc = []
    error_acc = []
    beta_acc = []
    beta_iter = beta
    counter = 0
    for i in tqdm(range(num_iter_mcmc), desc='MCMC iterations [' + str(num_iter_mcmc) + ']'):
        # if random_wait and np.random.choice(a=[0,1], p=[0.7, 0.3]):
        #     beta_iter = simulated_annealing(beta, i, schedule_function)
        # elif not random_wait:
        #     beta_iter = simulated_annealing(beta, i, schedule_function)
        if not counter:
            counter = np.random.choice(a=np.arange(50,150))
            beta_iter = simulated_annealing(beta, i, schedule_function)
        else:
            counter -= 1
        energy, error, X_vector = transition_function(Y_vector, X_vector_true, X_vector, W_matrix, vector_size, beta_iter)
        beta_acc.append(beta_iter)
        energy_acc.append(energy)
        error_acc.append(error)
    return beta_acc, energy_acc, error_acc, X_vector

def get_average_statistics(num_samples, vector_size, beta, transition_function, schedule_function, random_wait, num_iter_mcmc, num_exp):
    '''Run multiple experiments to average the statistics.'''
    beta_acc = []
    energy_acc = []
    error_acc = []
    error_final_acc = []
    for i in tqdm(range(num_exp), desc='experiments [' + str(num_exp) + ']'):
        W_matrix = np.random.normal(size=(num_samples, vector_size))
        X_vector_true = np.random.choice(a=[-1,1], size=(vector_size,1))
        Y_vector = relu(np.matmul(W_matrix, X_vector_true) / np.sqrt(vector_size))
        beta_updated, energy, error, X_vector = mcmc(Y_vector, X_vector_true, W_matrix, vector_size, beta, transition_function, schedule_function, random_wait, num_iter_mcmc)
        beta_acc.append(beta_updated)
        energy_acc.append(energy)
        error_acc.append(error)
        error_final_acc.append(get_reconstructed_error(X_vector_true, X_vector, vector_size))
    return np.mean(beta_acc, 0), np.mean(energy_acc, 0), np.mean(error_acc, 0), np.mean(error_final_acc, 0), np.std(error_final_acc, 0), 

def generate_data(N, alpha_array, beta_array, transition_function, schedule_function, random_wait, num_iter_mcmc, num_exp):
    '''Generate and save the data from the experiments.'''
    data = []
    for i, alpha in enumerate(tqdm(alpha_array, desc='alpha ' + str(alpha_array))):
        for j, beta in enumerate(tqdm(beta_array, desc='beta ' + str(beta_array))):
            beta_mean, energy_mean, error_mean, error_final_mean, error_final_std = get_average_statistics(int(alpha*N), N, beta, transition_function, schedule_function, random_wait, num_iter_mcmc, num_exp)
            data.append({'alpha':alpha, 'beta':beta, 'beta_mean':beta_mean, 'energy_mean':energy_mean, 'error_mean':error_mean, 'error_final_mean':error_final_mean, 'error_final_std':error_final_std})
    
    filename = str(N) + '_' + str(alpha_array)+ '_' + str(beta_array) + '_' + str(transition_function.__name__) + '_' + str(schedule_function.__name__) +'_' + str(random_wait) + '_' + str(num_iter_mcmc) + '_' + str(num_exp)+ '.npy'
    np.save(filename, data)

def plot_energy(data, len_alpha_array, len_beta_array, filename):
    '''Plot the energy of the grid search on the parameters alpha and beta.'''
    fig, ax = plt.subplots(len_alpha_array,len_beta_array, sharex=True, sharey=True, figsize=(15,15))
    for k, d in enumerate(data):
        i = int(np.floor(k / len_beta_array))
        j = k % len_beta_array
        ax[i,j].plot(d['energy_mean'], color='b')
        ax[i,j].set_title(r'$\alpha = $' + str(d['alpha']) + r' and $\beta = $' + str(d['beta']))
        ax[i,j].text(2000, 1000, 'last value = {:.2f}'.format(d['energy_mean'][-1]))
        if j == 0:
            ax[i,j].set_ylabel('energy')
        if i == len_alpha_array-1:
            ax[i,j].set_xlabel('iteration')
    plt.tight_layout()
    fig.savefig(filename + '.png')

def plot_error(data, data_random, field, alpha_array, beta_array, filename):
    '''Plot the mean or standard deviation error for the different values of alpha and beta. It also compares the results with a random guess.'''
    if field == 'error_final_mean':
        label = 'error mean'
    elif field == 'error_final_std':
        label = 'error std'
    fig, ax = plt.subplots(1, 1, figsize=(15,15))

    # reconstruct data structure
    beta_array_rec = np.zeros((len(beta_array), len(alpha_array)))
    for k, d in enumerate(data):
        i = int(np.floor(k / len(beta_array)))
        j = k % len(beta_array)
        beta_array_rec[j, i] = d[field]
        
    for i, b in enumerate(beta_array_rec):
        ax.plot(alpha_array, b, color='b', label=r'$\beta = $' + str(beta_array[i]))
        ax.set_ylabel(label)
        ax.set_xlabel('alpha')

    # reconstruct data_random structure
    beta_array_rec = np.zeros((1, len(alpha_array)))
    for k, d in enumerate(data_random):
        beta_array_rec[0, k] = d[field]
    
    for i, b in enumerate(beta_array_rec):
        ax.plot(alpha_array, b, color='b', label='random', linestyle='--')
        ax.set_ylabel(label)
        ax.set_xlabel('alpha')

    plt.xticks(alpha_array)

    # for nice colors
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0.1, 0.9,len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])
    plt.legend()
    fig.savefig(filename + '.png')

def plot_error_and_schedule(data, len_alpha_array, filename):
    '''Plot the error with the schedule for specific alpha values.'''
    fig, ax_left = plt.subplots(len_alpha_array, 1, sharex=True, sharey=True, figsize=(15,15))
    for k, d in enumerate(data):
        ax_right = ax_left[k].twinx()
        ax_left[k].plot(d['energy_mean'], color='b', label=r'e($\hat{x}$, X)')
        ax_right.plot(d['beta_mean'], color='r', label=r'$\beta$')
        ax_left[k].set_title(r'$\alpha = $' + str(d['alpha']) + r' and $\beta_0 = $' + '{:.2f}'.format(d['beta']))
        #ax[k].text(2000, 1000, 'last value = {:.2f}'.format(d['energy_mean'][-1]))
        ax_right.tick_params('y', colors='r')
        ax_left[k].tick_params('y', colors='b')
        ax_right.legend(loc=1)
        ax_left[k].legend(loc=2)

        if k == len(data)-1:
            ax_left[k].set_xlabel('iteration')
            
    plt.tight_layout()
    fig.savefig(filename + '.png')

if __name__ == '__main__':

    # parameters
    N = 1000
    num_iter_mcmc = 8000
    num_exp = 1
    alpha_array = np.linspace(0.5, 5, 10)
    # beta_array = np.linspace(0.5, 3, 6)
    beta_array = [1/(alpha * N) for alpha in alpha_array]

    # generate_data(N, alpha_array, beta_array, transition_function=metropolis_transition, schedule_function=constant_schedule, random_wait=False, num_iter_mcmc, num_exp)
    # data_metropolis_cst = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[0.5 1.  1.5 2.  2.5 3. ]_metropolis_transition_constant_schedule_False_8000_10.npy')
    # plot_energy(data_metropolis_cst, len(alpha_array), len(beta_array), 'energy_mean_grid_search_metropolis')

    generate_data(N, alpha_array, beta_array, transition_function=glauber_transition, schedule_function=constant_schedule, random_wait=False, num_iter_mcmc=num_iter_mcmc, num_exp=num_exp)
    data_glauber_cst = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[0.5 1.  1.5 2.  2.5 3. ]_glauber_transition_constant_schedule_False_8000_1.npy')
    plot_energy(data_glauber_cst, len(alpha_array), len(beta_array), 'energy_mean_grid_search_glauber')

    # generate_data(N, alpha_array, [0], transition_function=random_transition, schedule_function=constant_schedule, random_wait=False, num_iter_mcmc, num_exp)
    # data_random = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[0]_random_transition_8000_10.npy')

    # plot_error(data_metropolis_cst, data_random, 'error_final_mean', alpha_array, beta_array, 'error_mean')
    # plot_error(data_metropolis_cst, data_random, 'error_final_std', alpha_array, beta_array, 'error_std')

    alpha_array_restrained = np.linspace(3, 5, 3)
    # generate_data(N, alpha_array_restrained, [0.5518027258816396], metropolis_transition, linear_schedule, False, num_iter_mcmc, num_exp)
    # data_metropolis_linear = np.load('500_[3. 4. 5.]_[0.5518027258816396]_metropolis_transition_linear_schedule_False_8000_1.npy')
    # plot_error_and_schedule(data_metropolis_linear, len(alpha_array_restrained), 'error_mean_metropolis_linear_random_wait_False_1_exp')

    # generate_data(N, alpha_array_restrained, [1], metropolis_transition, exponential_schedule, False, num_iter_mcmc, num_exp)
    #data_metropolis_exponential = np.load('500_[3. 4. 5.]_[1]_metropolis_transition_exponential_schedule_False_8000_2.npy')
    #plot_error_and_schedule(data_metropolis_exponential, len(alpha_array_restrained), 'error_mean_metropolis_exponential_random_wait_False_1_exp')
