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
    '''Return the reconstruction error from the estimate and the ground truth.'''
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
    step = 500/8000
    return beta + step*t

def exponential_schedule(beta, t):
    '''Implementation of an exponential schedule for simulated annealing.'''
    e = np.exp(np.log(5)/8000)
    return beta * e**t

def simulated_annealing(beta, t, schedule_function):
    '''Implementation of the simulated annealing method.'''
    return schedule_function(beta, t)

def mcmc(Y_vector, X_vector_true, W_matrix, vector_size, beta, transition_function, schedule_function, num_iter_mcmc):
    '''Apply the Markov Chain Monte-Carlo (MCMC) method.'''

    X_vector = np.random.choice(a=[-1,1], size=(vector_size,1))
    energy_acc = []
    error_acc = []
    beta_acc = []
    beta_iter = beta
    counter = 0
    for i in tqdm(range(num_iter_mcmc), desc='MCMC iterations [' + str(num_iter_mcmc) + ']'):
        if not counter:
            counter = np.random.choice(a=[0,0])#np.arange(N/2,3*N/2))
            beta_iter = simulated_annealing(beta, i, schedule_function)
        else:
            counter -= 1

        # apply the transition function
        energy, error, X_vector = transition_function(Y_vector, X_vector_true, X_vector, W_matrix, vector_size, beta_iter)
        beta_acc.append(beta_iter)
        energy_acc.append(energy)
        error_acc.append(error)

    return beta_acc, energy_acc, error_acc, X_vector

def mcmc_optimized(Y_vector, X_vector_true, W_matrix, vector_size, beta, transition_function, schedule_function):
    '''Apply the Markov Chain Monte-Carlo (MCMC) method with stop condition.'''

    X_vector = np.random.choice(a=[-1,1], size=(vector_size,1))
    energy_acc = []
    error_acc = []
    beta_acc = []
    beta_iter = beta
    best_error = (0, vector_size)
    patience = 3*vector_size
    t = 0
    while True:
        # wait to go through 2N iterations before updating the beta parameter
        if (t+1) % (2*vector_size) == 0:
            beta_iter = simulated_annealing(beta, t, schedule_function)

        # apply the transition function
        energy, error, X_vector = transition_function(Y_vector, X_vector_true, X_vector, W_matrix, vector_size, beta_iter)
        beta_acc.append(beta_iter)
        energy_acc.append(energy)
        error_acc.append(error)

        # keep track of the best error and its corresponding iteration t
        if best_error[1] > error:
            best_error = (t, error)
        
        # quit the process if we managed to get a perfect reconstruction or if the patience time has exceeded
        # if t % 200 == 0:
        if error == 0 or t-best_error[0] > patience:

            print(error, best_error)
            break
        
        # update iteration t
        t += 1

        #TODO check that every array has same length
    return beta_acc, energy_acc, error_acc, X_vector

def get_average_statistics(num_samples, vector_size, beta, transition_function, schedule_function, num_iter_mcmc, num_exp):
    '''Run multiple experiments to average the statistics.'''
    beta_acc = []
    energy_acc = []
    error_acc = []
    error_final_acc = []
    for i in tqdm(range(num_exp), desc='experiments [' + str(num_exp) + ']'):
        W_matrix = np.random.normal(size=(num_samples, vector_size))
        X_vector_true = np.random.choice(a=[-1,1], size=(vector_size,1))
        Y_vector = relu(np.matmul(W_matrix, X_vector_true) / np.sqrt(vector_size))
        beta_updated, energy, error, X_vector = mcmc(Y_vector, X_vector_true, W_matrix, vector_size, beta, transition_function, schedule_function, num_iter_mcmc)
        beta_acc.append(beta_updated)
        energy_acc.append(energy)
        error_acc.append(error)
        error_final_acc.append(get_reconstructed_error(X_vector_true, X_vector, vector_size))
    return np.mean(beta_acc, 0), np.mean(energy_acc, 0), np.mean(error_acc, 0), np.mean(error_final_acc, 0), np.std(error_final_acc, 0), 

def generate_data(N, alpha_array, beta_array, transition_function, schedule_function, num_iter_mcmc, num_exp):
    '''Generate and save the data from the experiments by creating all combinations of alpha and beta.'''
    data = []
    for alpha in tqdm(alpha_array, desc='alpha ' + str(alpha_array)):
        for beta in tqdm(beta_array, desc='beta ' + str(beta_array)):
            beta_mean, energy_mean, error_mean, error_final_mean, error_final_std = get_average_statistics(int(alpha*N), N, beta, transition_function, schedule_function, num_iter_mcmc, num_exp)
            data.append({'alpha':alpha, 'beta':beta, 'beta_mean':beta_mean, 'energy_mean':energy_mean, 'error_mean':error_mean, 'error_final_mean':error_final_mean, 'error_final_std':error_final_std})
    
    filename = str(N) + '_' + str(alpha_array)+ '_' + str(beta_array) + '_' + str(transition_function.__name__) + '_' + str(schedule_function.__name__) + '_' + str(num_iter_mcmc) + '_' + str(num_exp)+ '.npy'
    np.save(filename, data)

def load_data(data, len_alpha_array, len_beta_array, list_tuple):
    '''Load the data of interest characterized by a list of alpha and beta tuple.'''
    res = []
    for i, d in enumerate(data):
        if (d['alpha'], d['beta']) in list_tuple:
            res.append(d)
    return res

def plot_energy(data, len_alpha_array, len_beta_array, filename):
    '''Plot the energy of the grid search on the parameters alpha and beta.'''
    fig, ax = plt.subplots(len_alpha_array,len_beta_array, sharex=True, sharey=True, figsize=(15,15))
    N = len(data[0]['energy_mean'])
    for k, d in enumerate(data):
        i = int(np.floor(k / len_beta_array))
        j = k % len_beta_array
        alpha = d['alpha']
        ax[i,j].plot([e/(alpha*N) for e in d['energy_mean']], color='b')
        ax[i,j].set_title(r'$\alpha = $' + str(alpha) + r' and $\beta = $' + str(d['beta']))
        if d['energy_mean'][-1] == 0:
            ax[i,j].plot(list(d['energy_mean']).index(0), 0, marker='o', color='b')
        # if j == 0:
        #     ax[i,j].set_ylabel('energy')
        # if i == len_alpha_array-1:
        #     ax[i,j].set_xlabel('iteration')

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel('\niterations', size=16)
    plt.ylabel('energy\n\n', size=16)
    plt.tight_layout()
    fig.savefig(filename + '.pdf')

def plot_error(data, data_random, alpha_array, beta_array, filename):
    '''Plot the mean or standard deviation error for the different values of alpha and beta. It also compares the results with a random guess.'''
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0.1, 0.9, len(beta_array) + 1)]

    # reconstruct data structure
    mean_array = np.zeros((len(beta_array), len(alpha_array)))
    std_array = np.zeros((len(beta_array), len(alpha_array)))
    for k, d in enumerate(data):
        i = int(np.floor(k / len(beta_array)))
        j = k % len(beta_array)
        mean_array[j, i] = d['error_final_mean']
        std_array[j, i] = d['error_final_std']
        
    for i, b in enumerate(mean_array):
        ax.errorbar(alpha_array, b, std_array[i], color=colors[i], capsize=3, label=r'$\beta = $' + str(beta_array[i]))
        ax.set_ylabel('error mean and std\n')
        ax.set_xlabel('\nalpha')

    # reconstruct data_random structure
    mean_array_random = np.zeros((1, len(alpha_array)))
    std_array_random = np.zeros((1, len(alpha_array)))
    for k, d in enumerate(data_random):
        mean_array_random[0, k] = d['error_final_mean']
        std_array_random[0, k] = d['error_final_std']
    
    for i, b in enumerate(mean_array_random):
        ax.errorbar(alpha_array, b, std_array_random[i], fmt='--', color=colors[-1], capsize=3, label='random', linestyle='--')

    plt.xticks(alpha_array)
    plt.legend()
    plt.tight_layout()
    fig.savefig(filename + '.pdf')

def plot_error_and_schedule(data_cst, len_alpha_array, len_beta_array, list_tuple, data, len_alpha_array_sa, filename):
    '''Plot the error with the schedule for specific alpha values.'''

    data_original = load_data(data_cst, len_alpha_array, len_beta_array, list_tuple)

    fig, ax_left = plt.subplots(len_alpha_array_sa, 1, sharex=True, sharey=True, figsize=(10,10))
    for k, d in enumerate(data):
        ax_right = ax_left[k].twinx()
        ax_left[k].plot(d['error_mean'], color='b', label='linear schedule')
        ax_left[k].plot(data_original[k]['error_mean'], color='b', linestyle=':', label=r'cst schedule $\beta = $' + '{:.1f}'.format(data_original[k]['beta']))
        ax_right.plot(d['beta_mean'], color='r', label=r'$\beta$')
        ax_left[k].set_title(r'$\alpha = $' + str(d['alpha']) + r' and $\beta_0 = $' + '{:.1f}'.format(d['beta']))
        #ax[k].text(2000, 1000, 'last value = {:.2f}'.format(d['energy_mean'][-1]))
        ax_right.tick_params('y', colors='r')
        ax_left[k].tick_params('y', colors='b')
        ax_right.legend(loc=1)
        ax_left[k].legend(loc=2)

        if k == len(data)-1:
            ax_left[k].set_xlabel('iteration')
            
    plt.tight_layout()
    fig.savefig(filename + '.pdf')

if __name__ == '__main__':

    # fixed parameters
    N = 500
    NUM_ITER_MCMC = 8000
    NUM_EXP = 10
    ALPHA_ARRAY = np.linspace(0.5, 5, 10) 
    BETA_ARRAY = np.linspace(3.5, 5, 4)

    # generate_data(N=N, alpha_array=ALPHA_ARRAY, beta_array=BETA_ARRAY, transition_function=metropolis_transition, schedule_function=constant_schedule, num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP)
    data_metropolis_cst = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[0.5 1.  1.5 2.  2.5 3. ]_metropolis_transition_constant_schedule_False_8000_10.npy')
    # plot_energy(data_metropolis_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), 'normalized_energy_mean_grid_search_metropolis')
    # data_metropolis_cst = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[3.5 4.  4.5 5. ]_metropolis_transition_constant_schedule_8000_10.npy')
    # plot_energy(data_metropolis_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), 'normalized_energy_mean_grid_search_metropolis_sequel')

    # generate_data(N=N, alpha_array=ALPHA_ARRAY, beta_array=BETA_ARRAY, transition_function=glauber_transition, schedule_function=constant_schedule, num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP)
    data_glauber_cst = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[0.5 1.  1.5 2.  2.5 3. ]_glauber_transition_constant_schedule_8000_10.npy')
    # plot_energy(data_glauber_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), 'normalized_energy_mean_grid_search_glauber')
    # data_glauber_cst = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[3.5 4.  4.5 5. ]_glauber_transition_constant_schedule_8000_10.npy')
    # plot_energy(data_glauber_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), 'normalized_energy_mean_grid_search_glauber_sequel')

    # generate_data(N=N, alpha_array=ALPHA_ARRAY, beta_array=[0], transition_function=random_transition, schedule_function=constant_schedule, num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP)
    # data_random = np.load('500_[0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]_[0]_random_transition_constant_schedule_8000_10.npy')

    # plot_error(data=data_metropolis_cst, data_random=data_random, alpha_array=ALPHA_ARRAY, beta_array=BETA_ARRAY, filename='error_mean_std_metropolis')
    # plot_error(data=data_glauber_cst, data_random=data_random, alpha_array=ALPHA_ARRAY, beta_array=BETA_ARRAY, filename='error_mean_std_glauber')

    ALPHA_ARRAY_SA = np.linspace(3, 5, 3)
    BETA_0 = 0.5
    # generate_data(N=N, alpha_array=ALPHA_ARRAY_SA, beta_array=[BETA_0], transition_function=metropolis_transition, schedule_function=linear_schedule, num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP)
    # data_metropolis_linear = np.load('500_[3. 4. 5.]_[0.5]_metropolis_transition_linear_schedule_8000_10.npy')
    # plot_error_and_schedule(data_metropolis_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), list(zip(ALPHA_ARRAY_SA, [3.0,2.0,3.0])), data_metropolis_linear, len(ALPHA_ARRAY_SA), 'error_mean_std_metropolis_linear')

    # generate_data(N=N, alpha_array=ALPHA_ARRAY_SA, beta_array=[BETA_0], transition_function=glauber_transition, schedule_function=linear_schedule, num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP)
    # data_glauber_linear = np.load('500_[3. 4. 5.]_[0.5]_glauber_transition_linear_schedule_8000_10.npy')
    # plot_error_and_schedule(data_glauber_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), list(zip(ALPHA_ARRAY_SA, [3.0,2.0,3.0])), data_glauber_linear, len(ALPHA_ARRAY_SA), 'error_mean_std_glauber_linear')

    generate_data(N=N, alpha_array=ALPHA_ARRAY_SA, beta_array=[BETA_0], transition_function=metropolis_transition, schedule_function=exponential_schedule, num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP)
    data_metropolis_exp = np.load('500_[3. 4. 5.]_[0.5]_metropolis_transition_exponential_schedule_8000_10.npy')
    plot_error_and_schedule(data_metropolis_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), list(zip(ALPHA_ARRAY_SA, [3.0,2.0,3.0])), data_metropolis_exp, len(ALPHA_ARRAY_SA), 'error_mean_std_metropolis_exp')

    # generate_data(N=N, alpha_array=ALPHA_ARRAY_SA, beta_array=[BETA_0], transition_function=glauber_transition, schedule_function=exponential_schedule, num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP)
    # data_glauber_exp = np.load('500_[3. 4. 5.]_[0.5]_glauber_transition_exponential_schedule_8000_10.npy')
    # plot_error_and_schedule(data_glauber_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), list(zip(ALPHA_ARRAY_SA, [3.0,2.0,3.0])), data_glauber_exp, len(ALPHA_ARRAY_SA), 'error_mean_std_glauber_exp')


