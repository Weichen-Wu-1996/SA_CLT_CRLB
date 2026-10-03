import numpy as np
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
        model_name: name of the data-generating model
        model_param: parameters for the data-generating model
        d: dimension of the parameter
        eta0: initial stepsize
        alpha: stepsize decay speed
        save_t: list of saved iterations
    '''
    d = int(config['dim'])
    theta = np.zeros((d,1))

    alpha = config['alpha']
    assert alpha >= 0.5 and alpha <= 1, "alpha must be within [0.5,1]!"
    
    # Total number of iterations
    save_iter = list(config['save_iter'])
    if not save_iter or any(t < 1 for t in save_iter):
        raise ValueError("save_iter must contain positive iteration numbers")
    if len(set(save_iter)) != len(save_iter):
        raise ValueError("save_iter must not contain duplicates")
    T = max(save_iter) + 1
    n_save = len(save_iter)

    theta_bar = theta.copy()
    saved_theta_bars = np.zeros((n_save, d))

    model = models.MODEL_LIST[config['model']](config['model_param'], seed=seed)
    if model.d != d:
        raise ValueError(f"config dim ({d}) does not match model d ({model.d})")
    save_positions = {iteration: i for i, iteration in enumerate(save_iter)}

    for t in range(1,T):

        Xt = model.get_sample(t)
        G = model.get_G(Xt, theta)
        etat = config['eta0'] * t ** (-alpha)
        theta -= etat * G
        theta_bar += (theta - theta_bar) / t

        if t in save_positions:
            saved_theta_bars[save_positions[t]] = theta_bar.ravel()

    return saved_theta_bars


def _run_trial(config, seed):
    """Run one trial and retain its seed for deterministic output ordering."""
    return seed, stochastic_approximation(config, seed)


def SA_multi_trials(config):

    N_trials = config['N_trials']
    completed_trials = Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(_run_trial)(config, seed)
        for seed in range(N_trials)
    )
    results_by_seed = [None] * N_trials
    for seed, result in tqdm(
        completed_trials,
        total=N_trials,
        unit="trial",
        smoothing=0,
        dynamic_ncols=True,
    ):
        results_by_seed[seed] = result
    return np.stack(results_by_seed)
    

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
