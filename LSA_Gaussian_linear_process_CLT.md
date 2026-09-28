# Polyak--Ruppert CLT for Linear Stochastic Approximation Driven by a Gaussian Linear Process

## 1. Setting

Let

$$
\xi_t=
\begin{bmatrix}
\text{vec}(A_t)\\
b_t
\end{bmatrix}
=\psi^\star+u_t,
$$

where

$$
\psi^\star=
\begin{bmatrix}
\text{vec}(A^\star)\\
b^\star
\end{bmatrix}
$$

and $u_t$ is a causal stationary Gaussian linear process:

$$
u_t=\sum_{k=0}^{\infty}H_k\varepsilon_{t-k},
\qquad
\varepsilon_t\overset{\mathrm{i.i.d.}}{\sim}
\mathcal N(0,\Sigma_\varepsilon).
$$

We study the stochastic approximation recursion

$$
\theta_t
=\theta_{t-1}-\eta_t(A_t\theta_{t-1}-b_t),
\qquad
\eta_t=\eta_0t^{-\alpha},
$$

whose target is

$$
\theta^\star=(A^\star)^{-1}b^\star.
$$

The Polyak--Ruppert average is

$$
\bar\theta_T=\frac1T\sum_{t=1}^T\theta_t.
$$

## 2. Assumptions

We impose the following conditions.

1. **Stable mean dynamics.** The matrix $-A^\star$ is Hurwitz. Equivalently,
   there is a matrix $U\succ0$ and a constant $\lambda>0$ such that

$$
(A^\star)^\top U+UA^\star\succeq\lambda U.
$$

2. **Geometrically decaying linear filter.** There are constants $C>0$ and
   $\rho\in(0,1)$ such that

$$
\|H_k\|\leq C\rho^k,
\qquad k\geq0.
$$

3. **Nondegenerate innovations.** The innovation covariance satisfies
   $\Sigma_\varepsilon\succ0$.

4. **Nonsingular transfer function.** The transfer function

$$
H(z)=\sum_{k=0}^{\infty}H_kz^k
$$

   is nonsingular on the unit circle.

5. **Polynomial stepsizes.** The stepsize exponent satisfies

$$
\frac12<\alpha<1.
$$

These assumptions can be weakened, but they provide a convenient setting in
which the dependence and moment bounds are transparent.

## 3. Why uniform Lipschitz continuity fails

The estimating function is

$$
G(\xi_t,\theta)=A_t\theta-b_t.
$$

Therefore,

$$
\|G(\xi_t,\theta)-G(\xi_t,\theta')\|
\leq \|A_t\|\,\|\theta-\theta'\|.
$$

Because $A_t$ is Gaussian, $\|A_t\|$ is unbounded. There is no deterministic
uniform Lipschitz constant. On the other hand, Gaussianity guarantees

$$
\mathbb E\|A_t\|^q<\infty
$$

for every finite $q$. The uniform condition can consequently be replaced by
moment bounds for the random Lipschitz factor.

The linear-process representation also gives geometrically decaying physical
dependence. If the innovation $\varepsilon_{t-k}$ is replaced by an independent
copy, then the resulting change in $u_t$ has $L^q$ norm bounded by

$$
\delta_q(k)\leq C_q\|H_k\|\leq C_q\rho^k.
$$

This decay is the main tool for controlling dependence between $A_t$ and the
past SA iterates.

## 4. Main CLT

Define

$$
M=
\begin{bmatrix}
\theta^\star\otimes I_d & -I_d
\end{bmatrix}.
$$

The long-run covariance of $u_t$ is

$$
\Omega_\xi
=\sum_{h=-\infty}^{\infty}\text{Cov}(u_0,u_h)
=H(1)\Sigma_\varepsilon H(1)^\top,
$$

where

$$
H(1)=\sum_{k=0}^{\infty}H_k.
$$

### Theorem

Under the assumptions above, together with the iterate moment bounds stated
in Section 5, the Polyak--Ruppert average satisfies

$$
\sqrt T(\bar\theta_T-\theta^\star)
\xrightarrow{d}
\mathcal N(0,\Lambda_{\mathrm{CLT}}),
$$

where

$$
\boxed{
\Lambda_{\mathrm{CLT}}
=(A^\star)^{-1}
M\Omega_\xi M^\top
(A^\star)^{-\top}.
}
$$

## 5. Iterate moment bounds

Let

$$
\Delta_t=\theta_t-\theta^\star,
\qquad
\widetilde A_t=A_t-A^\star.
$$

Since

$$
A_t\theta^\star-b_t=Mu_t,
$$

the error recursion is

$$
\Delta_t
=\Delta_{t-1}
-\eta_t\left(
A^\star\Delta_{t-1}
+\widetilde A_t\Delta_{t-1}
+Mu_t
\right).
$$

The required stability statement is the following.

### Moment lemma

There is a constant $C<\infty$ such that, for all sufficiently large $t$,

$$
\mathbb E\|\Delta_t\|^2\leq C\eta_t,
\qquad
\mathbb E\|\Delta_t\|^4\leq C\eta_t^2.
$$

### Proof strategy

Choose a logarithmic blocking length

$$
\ell_t=\lceil c_0\log t\rceil
$$

and decompose

$$
u_t
=\sum_{k=0}^{\ell_t-1}H_k\varepsilon_{t-k}
+\sum_{k=\ell_t}^{\infty}H_k\varepsilon_{t-k}.
$$

The second term satisfies

$$
\left\|
\sum_{k=\ell_t}^{\infty}H_k\varepsilon_{t-k}
\right\|_{L^q}
\leq C_q\rho^{\ell_t},
$$

which can be made smaller than any prescribed polynomial power of $t^{-1}$.
The first term is independent of the history up to time $t-\ell_t$.

Using the Lyapunov norm induced by $U$, one obtains

$$
\|(I-\eta_tA^\star)v\|_U^2
\leq(1-c\eta_t)\|v\|_U^2
$$

for sufficiently large $t$. Comparing $\Delta_{t-1}$ with
$\Delta_{t-\ell_t}$, using the Gaussian moment bounds, and absorbing the
geometrically small tail gives

$$
\mathbb E\|\Delta_t\|_U^2
\leq
(1-c\eta_t)\mathbb E\|\Delta_{t-1}\|_U^2
+C\eta_t^2.
$$

A discrete Gronwall argument then yields

$$
\mathbb E\|\Delta_t\|^2=O(\eta_t).
$$

The same blocking argument applied to the fourth moment gives

$$
\mathbb E\|\Delta_t\|^4=O(\eta_t^2).
$$

## 6. Asymptotic linear representation

Rearranging the recursion gives

$$
A^\star\Delta_{t-1}
=\frac{\Delta_{t-1}-\Delta_t}{\eta_t}
-\widetilde A_t\Delta_{t-1}
-Mu_t.
$$

After summation,

$$
A^\star\frac1{\sqrt T}\sum_{t=1}^T\Delta_{t-1}
=B_T-R_T-rac1{\sqrt T}\sum_{t=1}^TMu_t,
$$

where

$$
B_T
=\frac1{\sqrt T}
\sum_{t=1}^T
\frac{\Delta_{t-1}-\Delta_t}{\eta_t}
$$

and

$$
R_T
=\frac1{\sqrt T}
\sum_{t=1}^T
\widetilde A_t\Delta_{t-1}.
$$

### 6.1 Boundary term

Abel summation gives

$$
B_T
=\frac1{\sqrt T}\left[
\eta_1^{-1}\Delta_0
-\eta_T^{-1}\Delta_T
+\sum_{t=1}^{T-1}
(\eta_{t+1}^{-1}-\eta_t^{-1})\Delta_t
\right].
$$

Since $\|\Delta_t\|_{L^2}=O(\eta_t^{1/2})$,

$$
\|B_T\|_{L^2}
=O\left(T^{(\alpha-1)/2}\right)
=o(1).
$$

### 6.2 Multiplicative remainder

The logarithmic blocking argument and the moment lemma give

$$
\|R_T-\mathbb ER_T\|_{L^2}
\lesssim T^{-\alpha/2}\log T
$$

and

$$
\|\mathbb ER_T\|
\lesssim
\frac1{\sqrt T}\sum_{t=1}^T\eta_t\log t
=O\left(T^{1/2-\alpha}\log T\right).
$$

Both quantities converge to zero because $\alpha\in(1/2,1)$. Thus

$$
R_T=o_{L^2}(1).
$$

It follows that

$$
\sqrt T(\bar\theta_T-\theta^\star)
=-(A^\star)^{-1}
\frac1{\sqrt T}\sum_{t=1}^TMu_t
+o_p(1).
$$

## 7. CLT for the Gaussian linear process

The normalized sum of $u_t$ is Gaussian for every $T$. Its covariance
converges to $\Omega_\xi$, so

$$
\frac1{\sqrt T}\sum_{t=1}^Tu_t
\xrightarrow{d}
\mathcal N(0,\Omega_\xi).
$$

Combining this limit with the asymptotic linear representation and applying
Slutsky's theorem proves the CLT in Section 4.

## 8. Cramer--Rao lower bound

Suppose the filter coefficients $H_k$ and innovation covariance
$\Sigma_\varepsilon$ are known and only the stationary mean $\psi^\star$ is
unknown. For a regular stationary Gaussian process, the asymptotic Fisher
information for the mean is

$$
\mathcal I_\psi=\Omega_\xi^{-1}.
$$

Differentiating

$$
\theta^\star=(A^\star)^{-1}b^\star
$$

gives

$$
\frac{\partial\theta^\star}{\partial\psi}
=-(A^\star)^{-1}M.
$$

Therefore, the asymptotic Cramer--Rao lower bound is

$$
\begin{aligned}
\Lambda_{\mathrm{CRLB}}
&=(A^\star)^{-1}
M\mathcal I_\psi^{-1}M^\top
(A^\star)^{-\top}\\
&=(A^\star)^{-1}
M\Omega_\xi M^\top
(A^\star)^{-\top}.
\end{aligned}
$$

Consequently,

$$
\boxed{
\Lambda_{\mathrm{CRLB}}=\Lambda_{\mathrm{CLT}}.
}
$$

Thus Polyak--Ruppert averaging remains asymptotically efficient even though
the observations are dependent and the estimating function is not uniformly
Lipschitz.

## 9. A genuinely infinite-memory example

To construct an observed process that is not AR($m$) for any finite $m$, let

$$
H_0=I,
\qquad
H_k=c\rho^{k^2}I,
\quad k\geq1,
$$

where $c>0$ is small enough that

$$
c\sum_{k=1}^{\infty}\rho^{k^2}<1.
$$

Then:

- the coefficients decay faster than geometrically;
- $H(z)$ is nonsingular on the unit circle;
- the transfer function is nonrational;
- the observed process has infinite memory and is not a finite-order AR
  process;
- the preceding moment and remainder arguments apply; and
- the Polyak--Ruppert covariance equals the asymptotic CRLB.

## 10. Remaining technical work

The displayed CLT follows once the moment lemma and multiplicative-remainder
bounds are established with complete constants. A publication-ready proof
should spell out the following steps in full:

1. the Lyapunov-norm contraction for the mean recursion;
2. the logarithmic blocking construction;
3. the second- and fourth-moment recursions;
4. the covariance bound for the centered multiplicative remainder;
5. the bias bound caused by dependence between $A_t$ and
   $\Delta_{t-1}$; and
6. the Toeplitz limit giving
   $\mathcal I_\psi=\Omega_\xi^{-1}$.

For comparison, averaged-SA CLTs under Markov dynamics are developed in
[G. Fort, *Central Limit Theorems for Stochastic Approximation with Controlled
Markov Chain Dynamics*](https://www.numdam.org/item/PS_2015__19__60_0/).
The proof proposed here replaces Markov drift and Poisson-equation estimates
with the geometric physical-dependence bounds of the linear process.
