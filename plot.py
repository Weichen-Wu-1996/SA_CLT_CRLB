from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import models


def _normal_density(x, variance):
    return np.exp(-0.5 * x ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def CLT_hist(data_dir):
    """Plot empirical SA errors against their asymptotic Gaussian limits.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing ``config.yaml`` and ``results.npy``.

    Returns
    -------
    fig, axes
        The Matplotlib figure and its two-dimensional array of axes.

    Notes
    -----
    ``config.yaml`` must contain ``plot_iter``. Set ``plot_crlb`` to true to
    add Gaussian densities using the CRLB covariance. The model must expose
    ``theta_star``, ``d``, ``calculate_clt_variance()``, and, when requested,
    ``calculate_crlb()``.
    """
    data_dir = Path(data_dir)
    with (data_dir / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)

    plot_iter = [int(t) for t in config["plot_iter"]]
    if not plot_iter:
        raise ValueError("plot_iter must contain at least one iteration")

    results = np.load(data_dir / "results.npy")
    model = models.MODEL_LIST[config["model"]](config["model_param"], seed=0)
    if results.ndim != 3 or results.shape[2] != model.d:
        raise ValueError(
            f"Expected results with shape (trials, saved iterations, {model.d}); "
            f"got {results.shape}"
        )

    save_positions = {int(t): i for i, t in enumerate(config["save_iter"])}
    missing = [t for t in plot_iter if t not in save_positions]
    if missing:
        raise ValueError(f"Requested plot iterations were not saved: {missing}")

    clt_covariance = model.calculate_clt_variance()
    plot_crlb = bool(config.get("plot_crlb", False))
    crlb_covariance = model.calculate_crlb() if plot_crlb else None

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(
        model.d,
        len(plot_iter),
        figsize=(4 * len(plot_iter), 3 * model.d),
        squeeze=False,
    )

    for column, iteration in enumerate(plot_iter):
        saved = results[:, save_positions[iteration], :]
        scaled_errors = np.sqrt(iteration) * (saved - model.theta_star)

        for dimension in range(model.d):
            ax = axes[dimension, column]
            values = scaled_errors[:, dimension]
            clt_variance = clt_covariance[dimension, dimension]
            standard_deviations = [np.sqrt(clt_variance)]
            if plot_crlb:
                standard_deviations.append(
                    np.sqrt(crlb_covariance[dimension, dimension])
                )

            lower, upper = np.quantile(values, [0.001, 0.999])
            bound = max(abs(lower), abs(upper), 4.25 * max(standard_deviations))
            x_grid = np.linspace(-bound, bound, 600)

            ax.hist(
                values,
                bins=55,
                range=(-bound, bound),
                density=True,
                color="#1f77b4",
                edgecolor="white",
                linewidth=0.25,
                label="Empirical",
            )
            ax.plot(
                x_grid,
                _normal_density(x_grid, clt_variance),
                color="red",
                linewidth=1.35,
                label="CLT",
            )
            if plot_crlb:
                ax.plot(
                    x_grid,
                    _normal_density(x_grid, crlb_covariance[dimension, dimension]),
                    color="#2ca02c",
                    linewidth=1.35,
                    linestyle="--",
                    label="CRLB",
                )

            exponent = int(np.log2(iteration))
            is_power_of_two = 2 ** exponent == iteration
            iteration_label = f"2^{{{exponent}}}" if is_power_of_two else str(iteration)
            coordinate = dimension + 1
            ax.set_xlim(-bound, bound)
            ax.set_xlabel(
                rf"$\sqrt{{T}}(\bar{{\theta}}_{{T,{coordinate}}}"
                rf"-\theta^\star_{{{coordinate}}}),\;T={iteration_label}$"
            )
            if column == 0:
                ax.set_ylabel("Probability density")
            if dimension == 0:
                ax.set_title(rf"$T={iteration_label}$")
            ax.legend(loc="upper right", frameon=True)
            ax.grid(False)

    fig.tight_layout(pad=1.1, w_pad=1.0, h_pad=1.2)
    output_name = config.get("plot_output", "clt_histograms.pdf")
    fig.savefig(data_dir / output_name, bbox_inches="tight")
    return fig, axes
