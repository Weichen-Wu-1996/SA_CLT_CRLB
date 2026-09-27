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
    """Linear stochastic approximation with i.i.d. Gaussian entry noise.

    Required ``model_param`` fields are ``A``, ``b``, and ``noise_std``. Each
    entry of A_t - A and b_t - b is independently N(0, noise_std**2).
    """

    def __init__(self, model_param, seed=None):
        self.A = np.asarray(model_param["A"], dtype=float)
        self.b = np.asarray(model_param["b"], dtype=float)
        self.noise_std = float(model_param["noise_std"])

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square matrix")
        self.d = self.A.shape[0]
        if self.b.shape != (self.d,):
            raise ValueError(f"b must have shape ({self.d},)")
        if not np.isfinite(self.A).all() or not np.isfinite(self.b).all():
            raise ValueError("A and b must contain only finite values")
        if not np.isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("noise_std must be a positive finite scalar")

        eigenvalues = np.linalg.eigvals(self.A)
        if np.any(eigenvalues.real <= 0):
            raise ValueError(
                "-A must be Hurwitz, so every eigenvalue of A must have "
                "strictly positive real part"
            )

        self.theta_star = np.linalg.solve(self.A, self.b)
        self.rng = np.random.default_rng(seed)
        self.parameter_dim = self.d * (self.d + 1)

    def get_sample(self, _t=None):
        """Draw an independent pair (A_t, b_t)."""
        A_t = self.A + self.rng.normal(
            scale=self.noise_std, size=(self.d, self.d)
        )
        b_t = self.b + self.rng.normal(scale=self.noise_std, size=self.d)
        return A_t, b_t

    def get_G(self, sample, theta):
        """Return the linear estimating function A_t theta - b_t."""
        A_t, b_t = sample
        theta = np.asarray(theta, dtype=float).reshape(self.d)
        return (A_t @ theta - b_t).reshape(self.d, 1)

    def get_score(self, sample):
        """Return the Gaussian location score with respect to (vec(A), b)."""
        A_t, b_t = sample
        centered_parameter = np.concatenate((
            (np.asarray(A_t) - self.A).reshape(-1, order="F"),
            np.asarray(b_t) - self.b,
        ))
        return centered_parameter / self.noise_std ** 2

    def get_td_jacobian(self):
        """Return J = A."""
        return self.A.copy()

    def get_fisher_information(self):
        """Return the Fisher information for one Gaussian observation."""
        return np.eye(self.parameter_dim) / self.noise_std ** 2

    def get_gradient_parameter_map(self):
        """Return L such that G* = L [vec(A_t-A); b_t-b]."""
        A_map = np.kron(self.theta_star.reshape(1, -1), np.eye(self.d))
        return np.hstack((A_map, -np.eye(self.d)))

    def get_gradient_long_run_variance(self):
        """Return Var(A_t theta_star - b_t); observations are i.i.d."""
        parameter_map = self.get_gradient_parameter_map()
        covariance = self.noise_std ** 2 * parameter_map @ parameter_map.T
        return (covariance + covariance.T) / 2

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
        """Return the i.i.d. Gaussian CRLB for T Var(theta_hat)."""
        jacobian = self.get_td_jacobian()
        fisher = self.get_fisher_information()
        derivative = self.get_gradient_parameter_map()
        middle = derivative @ np.linalg.solve(fisher, derivative.T)
        return self._sandwich(jacobian, middle)


class LR_AR:
    """Linear regression with stationary AR(m) noise from Example 1.3.

    Required ``model_param`` fields are ``d``, ``m``, ``alpha``, and
    ``sigma``. Here ``sigma`` is the innovation standard deviation in
    epsilon_t ~ N(0, sigma**2). ``theta_star`` may be supplied; otherwise it
    is generated once from the optional model ``seed``.
    """

    def __init__(self, model_param, seed=None):
        self.d = int(model_param["d"])
        self.m = int(model_param["m"])
        self.alpha = np.asarray(model_param["alpha"], dtype=float)
        self.sigma = float(model_param["sigma"])

        if self.d < 1:
            raise ValueError("d must be positive")
        if self.m < 1:
            raise ValueError("m must be positive")
        if self.alpha.shape != (self.m,):
            raise ValueError(f"alpha must contain exactly m={self.m} entries")
        if not np.isfinite(self.alpha).all():
            raise ValueError("alpha must contain only finite values")
        if self.alpha[-1] == 0:
            raise ValueError("alpha_m must be nonzero")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be a positive finite scalar")

        self.companion = np.zeros((self.m, self.m))
        self.companion[0] = self.alpha
        if self.m > 1:
            self.companion[1:, :-1] = np.eye(self.m - 1)
        companion_roots = np.linalg.eigvals(self.companion)
        if np.any(np.abs(companion_roots) >= 1):
            raise ValueError(
                "The AR coefficients are not stationary: all companion-matrix "
                "eigenvalues must have modulus less than one"
            )

        model_rng = np.random.default_rng(model_param.get("seed", 42))
        if "theta_star" in model_param:
            self.theta_star = np.asarray(model_param["theta_star"], dtype=float)
            if self.theta_star.shape != (self.d,):
                raise ValueError(f"theta_star must have shape ({self.d},)")
            if not np.isfinite(self.theta_star).all():
                raise ValueError("theta_star must contain only finite values")
        else:
            self.theta_star = model_rng.normal(size=self.d)

        self.rng = np.random.default_rng(seed)
        self.noise_state_covariance = self._stationary_state_covariance()
        self.noise_history = self.rng.multivariate_normal(
            np.zeros(self.m), self.noise_state_covariance
        )
        self.feature_history = self._sample_sphere(self.m)

    def _stationary_state_covariance(self):
        """Solve C = F C F^T + sigma^2 e_1 e_1^T."""
        innovation_covariance = np.zeros((self.m, self.m))
        innovation_covariance[0, 0] = self.sigma ** 2
        system = np.eye(self.m ** 2) - np.kron(
            self.companion, self.companion
        )
        covariance = np.linalg.solve(
            system, innovation_covariance.reshape(-1, order="F")
        ).reshape((self.m, self.m), order="F")
        return (covariance + covariance.T) / 2

    def _sample_sphere(self, size=None):
        """Draw uniformly from the unit sphere S^(d-1)."""
        shape = (self.d,) if size is None else (size, self.d)
        gaussian = self.rng.normal(size=shape)
        return gaussian / np.linalg.norm(gaussian, axis=-1, keepdims=True)

    def get_sample(self, _t=None):
        """Draw (x_t, y_t) and retain terms needed for the AR score."""
        feature = self._sample_sphere()
        innovation = float(self.rng.normal(scale=self.sigma))
        noise = float(self.alpha @ self.noise_history + innovation)
        filtered_feature = feature - self.alpha @ self.feature_history
        response = float(feature @ self.theta_star + noise)

        sample = (feature, response, innovation, filtered_feature)
        self.noise_history[1:] = self.noise_history[:-1]
        self.noise_history[0] = noise
        self.feature_history[1:] = self.feature_history[:-1]
        self.feature_history[0] = feature
        return sample

    def get_G(self, sample, theta):
        """Return the least-squares gradient (x_t^T theta - y_t)x_t."""
        feature, response = sample[:2]
        theta = np.asarray(theta, dtype=float).reshape(self.d)
        return ((feature @ theta - response) * feature).reshape(self.d, 1)

    def get_score(self, sample):
        """Return the conditional likelihood score at theta_star."""
        _, _, innovation, filtered_feature = sample
        return innovation * filtered_feature / self.sigma ** 2

    def get_td_jacobian(self):
        """Return J = E[x_t x_t^T] = I_d / d."""
        return np.eye(self.d) / self.d

    def get_fisher_information(self):
        """Return the asymptotic Fisher information per observation."""
        multiplier = (1.0 + self.alpha @ self.alpha) / (
            self.d * self.sigma ** 2
        )
        return multiplier * np.eye(self.d)

    def get_gradient_long_run_variance(self):
        """Return the long-run covariance of G*(X_t)."""
        noise_variance = self.noise_state_covariance[0, 0]
        return (noise_variance / self.d) * np.eye(self.d)

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
        """Return the asymptotic CRLB for T Var(theta_hat)."""
        return np.linalg.solve(self.get_fisher_information(), np.eye(self.d))


MODEL_LIST = {
    "MRP": MRP,
    "LSA_Gaussian": LSA_Gaussian,
    "LR_AR": LR_AR,
}
