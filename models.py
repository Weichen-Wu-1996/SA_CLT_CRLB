import numpy as np


class MRP:
    """Finite-state MRP from Proposition III.10."""

    def __init__(self, model_param, seed=None):
        self.S = int(model_param["S"])
        self.d = int(model_param["d"])
        self.gamma = float(model_param["gamma"])
        if self.S < 1 or self.d < 1:
            raise ValueError("S and d must be positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")

        # Keep the MRP fixed across trials; use the trial seed only for its
        # sampled trajectory.
        model_rng = np.random.default_rng(model_param.get("seed", 42))
        self.rng = np.random.default_rng(seed)
        self.theta_star = model_rng.normal(size=self.d)
        self.psi = model_rng.normal(size=self.S)
        self.Phi = model_rng.normal(size=(self.S, self.d))
        self.Phi /= np.linalg.norm(self.Phi, axis=1, keepdims=True)

        logits = np.outer(self.psi, self.Phi @ self.theta_star)
        logits -= logits.max(axis=1, keepdims=True)
        self.P = np.exp(logits)
        self.P /= self.P.sum(axis=1, keepdims=True)

        self.r = (self.Phi - self.gamma * self.P @ self.Phi) @ self.theta_star
        self.mu = self._stationary_distribution()
        self.state = int(self.rng.choice(self.S, p=self.mu))

    def _stationary_distribution(self):
        """Return mu satisfying mu P = mu and sum(mu) = 1."""
        system = np.vstack((self.P.T - np.eye(self.S), np.ones(self.S)))
        target = np.append(np.zeros(self.S), 1.0)
        mu = np.linalg.lstsq(system, target, rcond=None)[0]
        mu = np.clip(mu, 0.0, None)
        return mu / mu.sum()

    def get_sample(self, _t=None):
        previous_state = self.state
        self.state = int(self.rng.choice(self.S, p=self.P[previous_state]))
        return previous_state, self.state

    def get_G(self, sample, theta):
        """Return the TD(0) estimating function A_t theta - b_t."""
        previous_state, current_state = sample
        theta = np.asarray(theta, dtype=float).reshape(self.d)
        phi = self.Phi[previous_state]
        td_residual = (
            (phi - self.gamma * self.Phi[current_state]) @ theta
            - self.r[previous_state]
        )
        return (td_residual * phi).reshape(self.d, 1)


class LSA_Gaussian:
    def __init__(self, model_param, seed=None):
        pass


class LR_AR:
    def __init__(self, model_param, seed=None):
        pass


MODEL_LIST = {
    "MRP": MRP,
    "LSA_Gaussian": LSA_Gaussian,
    "LR_AR": LR_AR,
}
