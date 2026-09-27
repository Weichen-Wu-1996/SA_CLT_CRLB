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

    def get_score(self, sample):
        """Return the transition score with respect to psi at the true MRP."""
        previous_state, current_state = sample
        values = self.Phi @ self.theta_star
        score = np.zeros(self.S)
        score[previous_state] = (
            values[current_state] - self.P[previous_state] @ values
        )
        return score

    def get_td_jacobian(self):
        """Return J = E[phi(s)(phi(s) - gamma phi(s'))^T]."""
        state_weights = self.mu[:, None] * self.Phi
        return self.Phi.T @ state_weights - self.gamma * (
            state_weights.T @ self.P @ self.Phi
        )

    def get_fisher_information(self):
        """Return E[S* S*^T] for one stationary transition.

        The score coordinate associated with state s is nonzero only when the
        previous state is s, so the transition Fisher information is diagonal.
        """
        values = self.Phi @ self.theta_star
        conditional_mean = self.P @ values
        conditional_second_moment = self.P @ (values ** 2)
        conditional_variance = conditional_second_moment - conditional_mean ** 2
        return np.diag(self.mu * np.maximum(conditional_variance, 0.0))

    def get_gradient_long_run_variance(self):
        """Return the long-run covariance Gamma_tilde of G*(X_t).

        Proposition III.10 gives G* = -gamma Phi^T S*.  Since the transition
        scores are martingale differences, all nonzero-lag covariances vanish.
        """
        fisher = self.get_fisher_information()
        gamma_tilde = self.gamma ** 2 * self.Phi.T @ fisher @ self.Phi
        return (gamma_tilde + gamma_tilde.T) / 2

    @staticmethod
    def _sandwich(jacobian, middle):
        """Compute J^{-1} middle J^{-T} without forming J^{-1}."""
        left = np.linalg.solve(jacobian, middle)
        covariance = np.linalg.solve(jacobian, left.T).T
        return (covariance + covariance.T) / 2

    def calculate_clt_variance(self):
        """Return the asymptotic covariance of sqrt(T)(theta_bar-theta_star)."""
        return self._sandwich(
            self.get_td_jacobian(), self.get_gradient_long_run_variance()
        )

    def calculate_crlb(self):
        """Return the asymptotic Markov CRLB for T Var(theta_hat).

        This implements Equation (III.5) using the Moore-Penrose inverse of
        the single-transition Fisher information.
        """
        jacobian = self.get_td_jacobian()
        fisher = self.get_fisher_information()
        derivative = -self.gamma * self.Phi.T @ fisher
        middle = derivative @ np.linalg.pinv(fisher) @ derivative.T
        return self._sandwich(jacobian, middle)


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
