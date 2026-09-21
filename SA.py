import numpy as np
import random
from tqdm import tqdm
import yaml
import click
import os


def stochastic_approximation(model_config, algo_config, seed: int=42):
    '''
    Stochastic Approximation with Polyak-Ruppert average, polynomial decay stepsizes
    Inputs:
    model_config: model configuration, including
        dim: dimension of the parameter theta
        get_sample: pointer to function to generate random samples X
        get_G: pointer to function to calculate G(X; theta)
    algo_config: algorithm configuration, including
        eta0: initial stepsize
        alpha: stepsize decay speed
        save_t: list of saved iterations
    '''
    d = model_config['dim']
    theta = np.zeros((d,1))

    alpha = algo_config['alpha']
    assert alpha >= 0.5 and alpha <= 1, "alpha must be within [0.5,1]!"
    
    # set random seed
    random.seed(seed)
    
    # Total number of iterations
    T = max(save_iter) + 1
    n_save = len(save_iter)

    theta_bar = theta.copy()
    saved_theta_bars = np.zeros((n_save, MRP.d))

    for t in range(1,T):

        Xt = model_config.get_sample(t)
        G = model_config.get_G(Xt, theta)
        etat = algo_config['eta0'] * t ** (-alpha)
        theta -= etat * G
        theta_bar += (theta - theta_bar) / t

        if t in save_iter:
            i = save_iter.index(t)
            saved_theta_bars[i] = theta_bar

    return saved_theta_bars

    
    