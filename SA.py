import numpy as np
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import yaml
import click
import os
import models


def stochastic_approximation(config, seed: int=42):
    '''
    Stochastic Approximation with Polyak-Ruppert average, polynomial decay stepsizes
    Inputs:
    config: configuration, including
        model: the data-generating model
        d: dimension of the parameter
        eta0: initial stepsize
        alpha: stepsize decay speed
        save_t: list of saved iterations
    '''
    d = config['dim']
    theta = np.zeros((d,1))

    alpha = config['alpha']
    assert alpha >= 0.5 and alpha <= 1, "alpha must be within [0.5,1]!"
    
    # set random seed
    random.seed(seed)
    
    # Total number of iterations
    T = max(save_iter) + 1
    n_save = len(save_iter)

    theta_bar = theta.copy()
    saved_theta_bars = np.zeros((n_save, MRP.d))

    model = models[config['model']]

    for t in range(1,T):

        Xt = model.get_sample(t)
        G = model.get_G(Xt, theta)
        etat = config['eta0'] * t ** (-alpha)
        theta -= etat * G
        theta_bar += (theta - theta_bar) / t

        if t in save_iter:
            i = save_iter.index(t)
            saved_theta_bars[i] = theta_bar

    return saved_theta_bars

def SA_multi_trials(config):

    N_trials = config['N_trials']
    one_trial_results = Parallel(n_jobs=-1)(delayed(stochastic_approximation)(config, seed) for seed in tqdm(range(N_trials)))
    results = np.concat(one_trial_results)
    return results
    

@click.command()
@click.argument("data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(data_dir):
    # Load config
    with open(os.path.join(data_dir,"config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    results = SA_multi_trials(config)
    
    np.save(os.path.join(data_dir,'results.npy'), results)
    
if __name__ == "__main__":
    main()