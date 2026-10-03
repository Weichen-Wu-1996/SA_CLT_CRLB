# Numerical Simulations of Central Limit Theorems and Cramer-Rao Lower Bounds for Stochastic Approximation on Stationary Processes

## Setup and plot generation

The Python packages needed to run the simulations and generate plots are NumPy, Matplotlib, PyYAML, tqdm, joblib, and Click. Install them from the repository root:

```bash
python3 -m pip install numpy matplotlib pyyaml tqdm joblib click
```

To generate a plot from an experiment's existing `config.yaml` and `results.npy`, run:

```bash
python3 -c "from plot import CLT_hist; CLT_hist('results/02')"
```

Replace `results/02` with the experiment directory listed below. The plot is saved in that directory under the filename specified by `plot_output` in its config (PDF for the current configs). To regenerate all four plots:

```bash
python3 -c "from plot import CLT_hist; [CLT_hist(f'results/{i:02d}') for i in range(1, 5)]"
```

If simulation data needs to be generated first, run the following command, which writes `results.npy` in the selected directory:

```bash
python3 SA.py results/02
```

Alternatively, install Jupyter with `python3 -m pip install notebook` and run `jupyter notebook plot.ipynb` to generate the plots interactively using [plot.ipynb](plot.ipynb).

## Results

### Linear Stochastic Approximation with Gaussian noise

Config: [results/02/config.yaml](results/02/config.yaml). Plots: [PDF](results/02/lsa_gaussian_clt_histograms.pdf) and [PNG](results/02/lsa_gaussian_clt_histograms.png).

![](results/02/lsa_gaussian_clt_histograms.png)

### Markov reward process

Config: [results/01/config.yaml](results/01/config.yaml). Plots: [PDF](results/01/mrp_clt_histograms.pdf) and [PNG](results/01/mrp_clt_histograms.png).

![](results/01/mrp_clt_histograms.png)

### Linear Regression with auto-regressive noise

Config: [results/03/config.yaml](results/03/config.yaml). Plots: [PDF](results/03/lr_ar_clt_histograms.pdf) and [PNG](results/03/lr_ar_clt_histograms.png).

![](results/03/lr_ar_clt_histograms.png)

### Linear Stochastic Approximation with Gaussian linear-process noise

Config: [results/04/config.yaml](results/04/config.yaml). Plots: [PDF](results/04/lsa_gaussian_linear_process_clt_histograms.pdf) and [PNG](results/04/lsa_gaussian_linear_process_clt_histograms.png).

![](results/04/lsa_gaussian_linear_process_clt_histograms.png)
