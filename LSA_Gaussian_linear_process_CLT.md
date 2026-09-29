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

Under the assumptions above, together with either of the two moment-control
conditions stated at the end of Section 5, the Polyak--Ruppert average
satisfies

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

## 5. Iterate moment control: the first genuinely new step

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

### 5.1 What is already supplied by the current paper

The deterministic part of the argument is unchanged. In the Lyapunov norm
induced by $U$,

$$
\|(I-\eta_tA^\star)v\|_U^2
\leq(1-c\eta_t)\|v\|_U^2
$$

for all sufficiently large $t$. This is precisely the mean-recursion
contraction used in the current paper.

The logarithmic blocking idea also carries over. Set

$$
\ell_t=\lceil c_0\log t\rceil
$$

and decompose

$$
u_t
=\sum_{k=0}^{\ell_t-1}H_k\varepsilon_{t-k}
+\sum_{k=\ell_t}^{\infty}H_k\varepsilon_{t-k}.
$$

The tail satisfies, for every finite $q$,

$$
\left\|
\sum_{k=\ell_t}^{\infty}H_k\varepsilon_{t-k}
\right\|_{L^q}
\leq C_q\rho^{\ell_t},
$$

which can be made smaller than any prescribed polynomial power of $t^{-1}$.
The recent part is independent of the history through time $t-\ell_t$.
Thus the contraction and basic blocking construction are not the new
difficulty. The difference from the proof in the current paper is that the
random Lipschitz coefficient $\|A_t\|$ is unbounded.

### 5.2 Exact second-moment expansion

Write

$$
Z_t=\widetilde A_t,
\qquad
\zeta_t=Mu_t,
\qquad
F_t=I-\eta_tA^\star,
\qquad
W_t=Z_t\Delta_{t-1}+\zeta_t.
$$

Then

$$
\|\Delta_t\|_U^2
=\|F_t\Delta_{t-1}\|_U^2
-2\eta_t\langle F_t\Delta_{t-1},W_t\rangle_U
+\eta_t^2\|W_t\|_U^2. \tag{5.1}
$$

The first term contracts. For the other two terms, replace
$\Delta_{t-1}$ by $\Delta_{t-\ell_t}$, use independence for the recent
innovation block, and bound the filter tail by $\rho^{\ell_t}$. The remaining
increment is

$$
D_{t,\ell_t}
=\Delta_{t-1}-\Delta_{t-\ell_t}
=-\sum_{j=t-\ell_t+1}^{t-1}
\eta_j\bigl(A^\star\Delta_{j-1}+Z_j\Delta_{j-1}+\zeta_j\bigr). \tag{5.2}
$$

If the products in (5.2) have the required moments, the argument in the
current paper yields a recursion of the schematic form

$$
m_{2,t}
\leq (1-c\eta_t)m_{2,t-1}
+C\eta_t^2\{1+\ell_t^2(1+\overline m_{2,t})\}
+C\eta_t\rho^{\ell_t}, \tag{5.3}
$$

where $m_{2,t}=\mathbb E\|\Delta_t\|_U^2$ and $\overline m_{2,t}$ is a
maximum over the blocking window. Closing (5.3) gives a bound of the form

$$
m_{2,t}=O\bigl(\eta_t\log^r t\bigr) \tag{5.4}
$$

for a fixed $r$; the current paper obtains $r=2$ in its uniformly Lipschitz
setting. This polylogarithmic loss is sufficient for the CLT. A sharp
$O(\eta_t)$ bound is not needed.

### 5.3 Why a separate fourth-moment recursion does not close

The earlier version of this note claimed that the same argument immediately
gave $\mathbb E\|\Delta_t\|^4=O(\eta_t^2)$. That claim was too quick.
Expanding with $x=F_t\Delta_{t-1}$ and $y=-\eta_tW_t$ gives

$$
\|x+y\|_U^4
\leq \|x\|_U^4
+4\|x\|_U^2\langle x,y\rangle_U
+C\bigl(\|x\|_U^2\|y\|_U^2+\|y\|_U^4\bigr). \tag{5.5}
$$

After blocking, (5.5) contains quantities such as

$$
\mathbb E\bigl[\|Z_t\|^4\|D_{t,\ell_t}\|^4\bigr]. \tag{5.6}
$$

But an $L^4$ bound for (5.2) requires an $L^4$ bound for
$Z_j\Delta_{j-1}$. Without independence between these two factors, Hölder's
inequality gives

$$
\|Z_j\Delta_{j-1}\|_{L^4}
\leq \|Z_j\|_{L^8}\|\Delta_{j-1}\|_{L^8}. \tag{5.7}
$$

Thus a fourth-moment induction asks for an eighth moment. Repeating the same
argument creates a hierarchy of higher moments. Gaussianity supplies all
moments of $Z_t$, but it does not by itself decouple $Z_t$ from
$\Delta_{t-1}$. Consequently, a second- and fourth-moment proof alone is
circular.

### 5.4 Two valid ways to close the gap

There are two reasonable formulations of the missing lemma.

**Route A: a simultaneous moment hierarchy.** Prove that for every fixed
integer $q\geq1$ there are constants $C_q,r_q<\infty$ such that

$$
\mathbb E\|\Delta_t\|^{2q}
\leq C_q\eta_t^q\log^{r_q}t. \tag{5.8}
$$

This needs a random-coefficient stability argument for the product of the
matrices $I-\eta_t(A^\star+Z_t)$; applying Hölder separately at each moment
order is insufficient. One possible proof uses a single exponential
Lyapunov estimate, from which (5.8) follows for all fixed $q$.

**Route B: Gaussian localization.** Let

$$
\chi_K(x)=x\min\{1,K/\|x\|\},
\qquad
\varepsilon_t^{(K)}=\chi_K(\varepsilon_t),
\qquad
u_t^{(K)}=\sum_{k\geq0}H_k\varepsilon_{t-k}^{(K)}.
$$

Radial clipping preserves centering and independence of the innovations, and
geometric summability gives the deterministic envelope

$$
\sup_t\|u_t^{(K)}\|
\leq K\sum_{k\geq0}\|H_k\|.
$$

For a horizon $T$, take

$$
K_T=C_0\sqrt{\log T}.
$$

Gaussian tails imply, for fixed $q$ and a constant $c_q>0$,

$$
\left\|\max_{1\leq t\leq T}
\|u_t-u_t^{(K_T)}\|\right\|_{L^q}
\leq C_qT^{1/q}e^{-c_qK_T^2}. \tag{5.9}
$$

Thus the coupling error is smaller than any chosen negative power of $T$ when
$C_0$ is large enough. The localized causal process has a deterministic
Lipschitz envelope of order $\sqrt{\log T}$, so the existing Lyapunov and
blocking proof can be rerun up to time $T$, with additional logarithmic
factors. A complete localization lemma must track its constants uniformly in
$T$ and propagate (5.9) through the two SA recursions. This finite-horizon
triangular-array step is essential; simply conditioning on a high-probability
event would destroy the independence used by blocking.

Route B is likely the shorter path if the goal is only the CLT. Route A gives
a stronger unconditional moment theorem. In the remainder of this note,
“moment control” means either (5.8), or the corresponding finite-horizon
localized bounds plus an $o_p(T^{-1/2})$ coupling error. Establishing one of
these statements is the first substantial new proof obligation beyond the
current paper.

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
=B_T-R_T-\frac1{\sqrt T}\sum_{t=1}^TMu_t,
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

If $\|\Delta_t\|_{L^2}=O(\eta_t^{1/2}\log^{r/2}t)$, then

$$
\|B_T\|_{L^2}
=O\left(T^{(\alpha-1)/2}\log^{r/2}T\right)
=o(1).
$$

### 6.2 Multiplicative remainder

Subject to the moment control in Section 5, the logarithmic blocking argument
is intended to give

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
- the preceding moment and remainder arguments apply once one of the
  moment-control lemmas in Section 5 is proved; and
- the Polyak--Ruppert covariance equals the asymptotic CRLB.

## 10. Proof status and remaining technical work

The six items in the earlier draft do not have equal status.

| Item | Status after comparison with the current paper |
| --- | --- |
| 1. Lyapunov contraction | Already proved in the current paper and reusable without a substantive change. |
| 2. Logarithmic blocking | The construction and tail estimates are already present. They extend naturally to the causal linear-process representation. |
| 3. Iterate moments | This is the first real new burden. A fourth-moment recursion by itself does not close; Section 5.3 identifies the resulting moment hierarchy. One must prove either the simultaneous bound (5.8) or a uniform finite-horizon localization lemma. |
| 4. Centered multiplicative remainder | The current paper's covariance-blocking calculation should apply after Item 3 supplies the needed product bounds, with extra logarithmic factors allowed. |
| 5. Dependence bias | The same lag decomposition is useful, but its random-Lipschitz products also depend on Item 3. The target rate is $o(T^{-1/2})$ after summation. |
| 6. Gaussian Fisher information | This is separate from SA stability and requires the standard Toeplitz/inverse-covariance limit for the mean parameter. |

The next theorem to prove should therefore be the finite-horizon localization
lemma in Route B, or the stronger all-even-moments proposition in Route A.
Only after that lemma is available should the centered remainder and bias
bounds be presented as complete proofs.

For comparison, averaged-SA CLTs under Markov dynamics are developed in
[G. Fort, *Central Limit Theorems for Stochastic Approximation with Controlled
Markov Chain Dynamics*](https://www.numdam.org/item/PS_2015__19__60_0/).
The proof proposed here replaces Markov drift and Poisson-equation estimates
with the geometric physical-dependence bounds of the linear process.
