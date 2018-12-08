# mini-project for MCAA course

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import os

def relu(vector):
    '''Return the Rectifier Linear Unit (ReLU) function applied to a vector. It is defined as x-> max{0,x}.'''
    vector[vector < 0] = 0
    return vector

def matmul(X,Y):
    '''Dummy manual matrix multiplication.'''
    # iterate through rows of X
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]
    return result

def get_energy(Y_vector, vector, W_matrix, vector_dim):
    '''Return the energy of a vector compared to an observation vector.'''
    return np.linalg.norm(Y_vector - relu(np.matmul(W_matrix, vector) / np.sqrt(vector_dim)))**2

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
    step = 5/8000
    return beta + step*t

def exponential_schedule(beta, t):
    '''Implementation of an exponential schedule for simulated annealing.'''
    e = np.exp(np.log(4)/8000)
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
            counter = 0
            # counter = np.random.choice(a=np.arange(N/2,3*N/2))
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
    for i in tqdm(range(num_exp), desc='experiments [' + str(num_exp) + ']'):
        W_matrix = np.random.normal(size=(num_samples, vector_size))
        X_vector_true = np.random.choice(a=[-1,1], size=(vector_size,1))
        Y_vector = relu(np.matmul(W_matrix, X_vector_true) / np.sqrt(vector_size))
        beta_updated, energy, error, X_vector = mcmc(
            Y_vector,
            X_vector_true,
            W_matrix,
            vector_size,
            beta,
            transition_function,
            schedule_function,
            num_iter_mcmc
            )
        beta_acc.append(beta_updated)
        energy_acc.append(energy)
        error_acc.append(error)
    return np.mean(beta_acc, 0), np.mean(energy_acc, 0), np.mean(error_acc, 0), np.std(error_acc, 0) 

def generate_data_exploration(N, alpha_array, beta_array, transition_function, schedule_function, num_iter_mcmc, num_exp):
    '''Generate and save the data from the experiments by creating all combinations of alpha and beta for exploration.'''
    data = []
    for alpha in tqdm(alpha_array, desc='alpha ' + str(alpha_array)):
        for beta in tqdm(beta_array, desc='beta ' + str(beta_array)):
            beta_mean, energy_mean, error_mean, error_std = get_average_statistics(
                int(alpha*N),
                N, 
                beta,
                transition_function,
                schedule_function,
                num_iter_mcmc,
                num_exp
                )
            data.append({
                'alpha':alpha,
                'beta':beta, 'beta_mean':beta_mean,
                'energy_mean':energy_mean,
                'error_mean':error_mean, 'error_std':error_std
                })
    
    filename = (str(N) + '_' + str(num_iter_mcmc) + '_' + str(num_exp) + '_' + str(alpha_array) + '_' + 
               str(beta_array) + '_' + str(transition_function.__name__) + '_' + str(schedule_function.__name__) + '.npy')
    np.save(filename, data)

def generate_data_sa(N, alpha_array, transition_function, schedule_function, num_iter_mcmc, num_exp):
    '''Generate and save the data from the experiments for the simulated annealing technique.'''
    data = []
    for alpha in tqdm(alpha_array, desc='alpha ' + str(alpha_array)):
        # to normalize the energy
        beta_0 = 1/alpha
        beta_mean, energy_mean, error_mean, error_std = get_average_statistics(
            int(alpha*N),
            N,
            beta_0,
            transition_function,
            schedule_function,
            num_iter_mcmc,
            num_exp
            )
        data.append({
            'alpha':alpha,
            'beta':beta_0, 'beta_mean':beta_mean,
            'energy_mean':energy_mean,
            'error_mean':error_mean, 'error_std':error_std
            })
    
    filename = (str(N) + '_' + str(num_iter_mcmc) + '_' + str(num_exp) + '_' + str(alpha_array) + '_' +
               str(transition_function.__name__) + '_' + str(schedule_function.__name__) + '.npy')
    np.save(filename, data)

def generate_data_parallel(alpha_beta_tuple, N, transition_function, schedule_function, num_iter_mcmc, num_exp):
    '''Generate the data from the specified configuration for parallelization.'''
    alpha = alpha_beta_tuple[0]
    beta = alpha_beta_tuple[1]
    beta_mean, energy_mean, error_mean, error_std = get_average_statistics(
        int(alpha*N), N, beta, transition_function, schedule_function, num_iter_mcmc, num_exp
        )
    return {
        'alpha':alpha,
        'beta':beta, 'beta_mean':beta_mean,
        'energy_mean':energy_mean,
        'error_mean':error_mean, 'error_std':error_std
        }

def load_data(data, len_alpha_array, list_tuple):
    '''Load the data of interest characterized by a list of alpha and beta tuple.'''
    res = []
    for i, d in enumerate(data):
        if (d['alpha'], d['beta']) in list_tuple:
            res.append(d)
    return res

def plot_energy(data_metropolis_cst, data_glauber_cst, len_alpha_array, len_beta_array, filename):
    '''Plot the energy of the grid search on the parameters alpha and beta.'''
    fig, ax = plt.subplots(len_alpha_array,len_beta_array, sharex=True, sharey=True, figsize=(15,15))
    N = len(data_metropolis_cst[0]['energy_mean'])
    for k, d in enumerate(data_metropolis_cst):
        i = int(np.floor(k / len_beta_array))
        j = k % len_beta_array
        alpha = d['alpha']
        ax[i,j].plot([e/(alpha*N) for e in d['energy_mean']], color='b', label='MH')
        ax[i,j].plot([e/(alpha*N) for e in data_glauber_cst[k]['energy_mean']], color='g', label='Gd')
        ax[i,j].set_title(r'$\alpha = $' + str(alpha) + r' and $\beta = $' + str(d['beta']))
        if d['energy_mean'][-1] == 0:
            ax[i,j].plot(list(d['energy_mean']).index(0), 0, marker='o', color='b')
        if data_glauber_cst[k]['energy_mean'][-1] == 0:
            ax[i,j].plot(list(data_glauber_cst[k]['energy_mean']).index(0), 0, marker='o', color='g')

    plt.legend()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel('\niterations', size=14)
    plt.ylabel('energy\n\n', size=14)
    plt.tight_layout()
    fig.savefig(filename + '.pdf')

def plot_error(
        schedule_type_name,
        data_metropolis_cst, beta_metropolis_cst, data_metropolis,
        data_glauber_cst, beta_glauber_cst, data_glauber,
        alpha_array, filename):
    '''Plot the error with a performance comparison between Metropolis-Hastings and Glauber dynamic algorithm.'''

    data_metropolis_cst = load_data(data_metropolis_cst, alpha_array, list(zip(alpha_array, beta_metropolis_cst)))
    data_glauber_cst = load_data(data_glauber_cst, alpha_array, list(zip(alpha_array, beta_glauber_cst)))

    fig, axes = plt.subplots(len(alpha_array), 1, sharex=True, sharey=True, figsize=(10,15))
    for k, d in enumerate(data_metropolis):
        # plot metropolis data
        axes[k].plot(d['error_mean'], color='b', label='{} MH'.format(schedule_type_name))
        axes[k].fill_between(
            np.arange(len(d['error_mean'])),
            d['error_mean']+d['error_std'],
            d['error_mean']-d['error_std'],
            color='b',
            alpha=0.1
            )
        axes[k].plot(
            data_metropolis_cst[k]['error_mean'],
            color='b',
            linestyle=':',
            label=r'constant MH with $\beta = $' + '{:.1f}'.format(beta_metropolis_cst[k])
            )
        axes[k].fill_between(
            np.arange(len(data_metropolis_cst[k]['error_mean'])),
            data_metropolis_cst[k]['error_mean']+data_metropolis_cst[k]['error_std'],
            data_metropolis_cst[k]['error_mean']-data_metropolis_cst[k]['error_std'],
            color='b',
            alpha=0.1
            )
        # plot glauber data
        axes[k].plot(data_glauber[k]['error_mean'], color='g', label='{} Gd'.format(schedule_type_name))
        axes[k].fill_between(
            np.arange(len(data_glauber[k]['error_mean'])),
            data_glauber[k]['error_mean']+data_glauber[k]['error_std'],
            data_glauber[k]['error_mean']-data_glauber[k]['error_std'],
            color='g',
            alpha=0.1
            )
        axes[k].plot(
            data_glauber_cst[k]['error_mean'],
            color='g',
            linestyle=':',
            label=r'constant Gd with $\beta = $' + '{:.1f}'.format(beta_glauber_cst[k])
            )
        axes[k].fill_between(
            np.arange(len(data_glauber_cst[k]['error_mean'])),
            data_glauber_cst[k]['error_mean']+data_glauber_cst[k]['error_std'],
            data_glauber_cst[k]['error_mean']-data_glauber_cst[k]['error_std'],
            color='g',
            alpha=0.1
            )
        axes[k].set_title(r'$\alpha = $' + str(d['alpha']))
        axes[k].legend()

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel('\niterations', size=14)
    plt.ylabel('error\n\n', size=14)
    plt.tight_layout()
    fig.savefig(filename + '.pdf')

def plot_schedules(list_data_schedules, len_alpha_array, filename):
    '''Plot the schedule function for the beta parameter.'''
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8,6))
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0.1, 0.9, len_alpha_array)]

    for k, (name, data_schedule) in enumerate(list_data_schedules):
        axes[k].set_title(name + ' schedule')
        for l in range(len_alpha_array):
            axes[k].plot(data_schedule[l]['beta_mean'], color=colors[l], label=r'$\alpha$ =' + str(data_schedule[l]['alpha']))
            axes[k].grid(True, linestyle=':')
    
    plt.legend()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel('\niterations', size=14)
    plt.ylabel(r'$\beta$' + '\n', size=14)
    plt.tight_layout()
    fig.savefig(filename + '.pdf')

if __name__ == '__main__':

    # fixed parameters
    N = 500
    NUM_ITER_MCMC = 8000
    NUM_EXP = 10
    ALPHA_ARRAY = [0.5, 1.0, 1.5, 2, 3, 4] 
    BETA_ARRAY = np.linspace(0.5, 3, 6)

    #####################
    # exploration phase #
    #####################

    # generate the data
    generate_data_exploration(
        N=N,
        alpha_array=ALPHA_ARRAY,
        beta_array=BETA_ARRAY,
        transition_function=metropolis_transition,
        schedule_function=constant_schedule,
        num_iter_mcmc=NUM_ITER_MCMC, 
        num_exp=NUM_EXP
        )
    generate_data_exploration(
        N=N,
        alpha_array=ALPHA_ARRAY,
        beta_array=BETA_ARRAY,
        transition_function=glauber_transition,
        schedule_function=constant_schedule,
        num_iter_mcmc=NUM_ITER_MCMC,
        num_exp=NUM_EXP
        )
    generate_data_exploration(
        N=N,
        alpha_array=ALPHA_ARRAY,
        beta_array=[0],
        transition_function=random_transition,
        schedule_function=constant_schedule,
        num_iter_mcmc=NUM_ITER_MCMC, num_exp=NUM_EXP
        )
    
    # load the data
    data_metropolis_cst = np.load('500_8000_10_[0.5, 1.0, 1.5, 2, 3, 4]_[0.5 1.  1.5 2.  2.5 3. ]_metropolis_transition_constant_schedule.npy')
    data_glauber_cst = np.load('500_8000_10_[0.5, 1.0, 1.5, 2, 3, 4]_[0.5 1.  1.5 2.  2.5 3. ]_glauber_transition_constant_schedule.npy')
    data_random = np.load('500_8000_10_[0.5, 1.0, 1.5, 2, 3, 4]_[0]_random_transition_constant_schedule.npy')

    # plot results
    plot_energy(data_metropolis_cst, data_glauber_cst, len(ALPHA_ARRAY), len(BETA_ARRAY), 'energy')

    #######################
    # simulated annealing #
    #######################

    # generate the data
    generate_data_sa(
        N=N,
        alpha_array=ALPHA_ARRAY,
        transition_function=metropolis_transition,
        schedule_function=linear_schedule,
        num_iter_mcmc=NUM_ITER_MCMC,
        num_exp=NUM_EXP
        )
    generate_data_sa(
        N=N,
        alpha_array=ALPHA_ARRAY,
        transition_function=metropolis_transition,
        schedule_function=exponential_schedule,
        num_iter_mcmc=NUM_ITER_MCMC,
        num_exp=NUM_EXP
        )
    generate_data_sa(
        N=N,
        alpha_array=ALPHA_ARRAY,
        transition_function=glauber_transition,
        schedule_function=linear_schedule,
        num_iter_mcmc=NUM_ITER_MCMC,
        num_exp=NUM_EXP
        )
    generate_data_sa(
        N=N,
        alpha_array=ALPHA_ARRAY,
        transition_function=glauber_transition,
        schedule_function=exponential_schedule,
        num_iter_mcmc=NUM_ITER_MCMC,
        num_exp=NUM_EXP
        )

    # load the data
    data_metropolis_lin = np.load('500_8000_10_[0.5, 1.0, 1.5, 2, 3, 4]_metropolis_transition_linear_schedule.npy')
    data_metropolis_exp = np.load('500_8000_10_[0.5, 1.0, 1.5, 2, 3, 4]_metropolis_transition_exponential_schedule.npy')
    data_glauber_lin = np.load('500_8000_10_[0.5, 1.0, 1.5, 2, 3, 4]_glauber_transition_linear_schedule.npy')
    data_glauber_exp = np.load('500_8000_10_[0.5, 1.0, 1.5, 2, 3, 4]_glauber_transition_exponential_schedule.npy')
    
    # follow rule beta = 1/alpha for comparison with min value set to 0.5
    beta_cst = [2, 1, 0.5, 0.5, 0.5, 0.5]

    # plot schedules
    plot_schedules([('linear', data_metropolis_lin), ('exponential', data_metropolis_exp)], len(ALPHA_ARRAY), 'schedules')

    # plot results
    plot_error(
        'linear',
        data_metropolis_cst, beta_cst, data_metropolis_lin,
        data_glauber_cst, beta_cst, data_glauber_lin,
        ALPHA_ARRAY,
        'error_linear_sa'
        )
    plot_error(
        'exponential',
        data_metropolis_cst, beta_cst, data_metropolis_exp,
        data_glauber_cst, beta_cst, data_glauber_exp,
        ALPHA_ARRAY,
        'error_exponential_sa'
        )

    ##########################
    # optimization tentative #
    ##########################

    # pool = mp.Pool(processes=4)
    # prod_ab = partial(
    #     generate_data_parallel,
    #     N=N,
    #     transition_function=metropolis_transition,
    #     schedule_function=linear_schedule,
    #     num_iter_mcmc=NUM_ITER_MCMC,
    #     num_exp=NUM_EXP
    #     )
    # alpha_beta_tuple_list = list(zip(ALPHA_ARRAY_SA, [3.0,2.0,3.0]))
    # output = list(pool.imap(prod_ab, alpha_beta_tuple_list))#list(tqdm(pool.map(prod_ab, alpha_beta_tuple_list), total=3))
    # # output = list(map(prod_ab, alpha_beta_tuple_list))
    # print('output:', output)
    # pool.close()
    # pool.join()
    # filename = 'test_mp'
    # # np.save(filename, output)

    # further work: https://stackoverflow.com/questions/15414027/multiprocessing-pool-makes-numpy-matrix-multiplication-slower 
    # to understand why np.matmul won't work correctly with pool.imap by default and find alternatives