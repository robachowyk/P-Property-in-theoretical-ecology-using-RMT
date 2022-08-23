# Jiri Rohn's algorithm to check P-property of a $(N \times N)$ matrix using **REG**ular**I**ty/**SING**ularity of interval matrix
### J. Rohn wrote his algorithm on matlab; here is a traduction of his algorithm in python.

## How to use
- Open Terminal
- Change the current working directory to the location where you want to clone this repository
- Type `git clone https://github.com/[...]`
- Press enter
- Go to the location where you cloned the repository
- Open check_p_property.ipynb

## The algorithm

The work presented on check_p_property.ipynb illustrates the phase transition for the P-matrix property of $I - \frac{X_N}{\alpha \sqrt{N}}$ according to the value of $\alpha$. 

- the matrix $I$ is the $(N \times N)$ identity matrix
- the matrix $X_N$ is a $(N \times N)$ random matrix whose inputs follow a symmetric distribution, for e.g. $\mathcal N (0,1)$
- $\alpha$ is a parameter on which depend the P-property of the matrix

This investigation is useful in theoretical ecology especially. To investigate the phase transition, we check the P-property of the matrix $I - \frac{X_N}{\alpha \sqrt{N}}$ for 15 values of $\alpha$ spread around the theoretical value of the phase change. 
- We use $\alpha \in$ `np.linspace(0.001,2,15)` for non-hermitian matrix for which the phenomenon of behavior change is supposed to occur at $\alpha = 1$. 
- We use $\alpha \in$ `np.linspace(1,3,15)` for hermitian matrix for which the phenomenon of behavior change is supposed to occur at $\alpha = 2$.

We then produce a Montecarlo estimate (for each value of $\alpha$) of the probability of being a P-matrix or not based on 50 computations. The value $\frac{1}{\alpha}$ is called *interaction strength*.

The algorithm is supposed to run in polynomial time $\mathcal O (N^3)$, for e.g. on:
- non-hermitian matrix of size $(N \times N)$
- with a Montecarlo estimate over 50 computations
- for 15 values of $\alpha$ spread around the phase change
the algorithm run in $\mathcal O (N^3) \approx 0.05 \times N^3$ seconds ($\approx 15$ hours for $(100 \times 100)$ matrix). 
It is faster for hermitian matrices.

## Heuristic proof of Jiri Rohn's algorithm

The algorithm is supposed to consider an exhaustive list of methods to determine singularity/regularity of the interval matrix $[A_c - \Delta,A_c + \Delta]$ built from `regising(Ac, Delta)` inputs.

[Jiri Rohn's theorem 2](http://dx.doi.org/10.1007/s11590-011-0318-y) states that for $(A - I)$ non-singular, $A$ is a P-Matrix $\iff$ $[ (A-I)^{-1} (A + I) - I, (A-I)^{-1} (A + I) + I ]$ is regular, i.e. it does not contain any singular matrix $S$. This interval matrix takes the form $[A_c - \Delta,A_c + \Delta]$ with $A_c = (A-I)^{-1} (A + I)$ and $\Delta = I$. 

Our interval of interest is therefore this one: $[A_c - \Delta,A_c + \Delta]$ where $A_c$ and $\Delta$ are defined as above.

Therefore `pmatrix` program do:

```python 
def pmatrix(A):
    n = A.shape[0]
    I = np.eye(n)
    if np.linalg.matrix_rank(A - I) == n:
        B = np.linalg.inv(A - I) @ (A + I)
        S = regising(B, I)
    return (S is empty)
```

Indeed `regising` checks the **REG**ular**I**ty/**SING**ularity of the interval matrix by returning a matrix $S$ singular, if one has been found in the interval matrix (i.e. singular interval) or a value $S = [ \ ]$ empty if no singular matrix has been found in the interval (i.e. regular interval).

**Disjunction elimination:**

- ***singularity of  the midpoint matrix Ac***: 

The midpoint matrix $A_c$ of the interval matrix $[A_c \pm \Delta]$ is singular.

- ***singularity via diagonal condition***: 

[Theorem 2.1 from Oettli and Prager](https://doi.org/10.1137/S0895479896310743) $|A_C x| \leq \Delta |x|$ has a non trivial (i.e. non zero) solution $x$ $\iff$ the interval matrix $[A_c \pm \Delta]$ is singular.

- ***singularity via steepest determinant descent***: 

Finds SINGularity via DETerminant DESCent ???

- ***singularity as a by-product of Qz-matrices***: 

Finds REGularity SUFFicient CONDition via matrices QZ based on [Theorem 4.3](https://doi.org/10.1137/S0895479896313978) $[A_c \pm \Delta]$ is singular $\iff$ the linear programming problem $(\star)$ is unbounded for some $z \in \{ \pm 1 \}^n$. The algorithm is looking for bounded or unbounded solutions of $(\star) = \max \{ z^T \cdot x ; (A_c - \Delta D_z) \cdot x \leq 0, (A_c + \Delta D_z) \cdot x \geq 0, D_z \cdot x \geq 0 \}
- ***singularity via the main algorithm***: 

Loop on $\{ \pm 1 \}^n$ to [find the singular matrix](https://doi.org/10.1137/0614007) If $[A_c \pm \Delta]$ is singular, there exists $x, x' \neq 0$, $y, z \in \{ \pm 1 \}^n$ such that:

$(A_c - d D_y \Delta D_z) \cdot x = 0$

$(A_c - d D_y \Delta D_z)^T \cdot x' = 0$

$D_z \cdot x \geq 0$

$D_y \cdot x' \geq 0$

where $d = d(A_c, \Delta) \in [0,1]$

$d(A_c, \Delta) = \min \{ \delta \geq 0; [A_c \pm \delta \Delta] \text{ is singular} \} = \frac{1}{\underset{y, z \in \{ \pm 1 \}^n}{\max} \rho^{\mathbb{R}} \left[ {A_c}^{-1} D_y \Delta D_z \right]}$.

- ***regularity  via Beeck's condition***: 

[Corollary 3.2 from Beeck](https://doi.org/10.1137/S0895479896310743): for $A_c$ non singular, $\rho (|{A_C}^{-1}| \Delta) < 1 \implies [A_C \pm \Delta]$ is regular.

- ***regularity  via symmetrization***: 

[Sections 4 and 5 from Rex and Rohn](https://doi.org/10.1137/S0895479896310743)

$\lambda_{\max}(\Delta^T \Delta) < \lambda_{\min}({A_c}^T A_c) \implies [A_C \pm \Delta]$ is regular.

$\lambda_{\max}({A_c}^T A_c) \leq \lambda_{\min}(\Delta^T \Delta) \implies [A_C \pm \Delta]$ is singular.

${A_c}^T A_c - \| \Delta^T \Delta \| I$ positive definite $\implies [A_C \pm \Delta]$ is regular.

$\Delta^T \Delta - {A_c}^T A_c$ positive definite $\implies [A_C \pm \Delta]$ is singular.


- ***regularity via two Qz-matrices***: 

Finds REGularity SUFFicient CONDition via matrices QZ based on [Theorem 4.3](https://doi.org/10.1137/S0895479896313978) $[A_c \pm \Delta]$ is regular $\iff$ the linear programming problem $(\star)$ is bounded for all $z \in \{ \pm 1 \}^n$. The algorithm is looking for bounded or unbounded solutions of $(\star) = \max \{ z^T \cdot x ; (A_c - \Delta D_z) \cdot x \leq 0, (A_c + \Delta D_z) \cdot x \geq 0, D_z \cdot x \geq 0 \}$

- ***regularity  via the main algorithm*** (too expansive): 

Loop on $\{ \pm 1 \}^n$ [did not find any singular matrix](https://doi.org/10.1137/0614007) If $[A_c \pm \Delta]$ is singular, there exists $x, x' \neq 0$, $y, z \in \{ \pm 1 \}^n$ such that:

$(A_c - d D_y \Delta D_z) \cdot x = 0$

$(A_c - d D_y \Delta D_z)^T \cdot x' = 0$

$D_z \cdot x \geq 0$

$D_y \cdot x' \geq 0$

where $d = d(A_c, \Delta) \in [0,1]$

$d(A_c, \Delta) = \min \{ \delta \geq 0; [A_c \pm \delta \Delta] \text{ is singular} \} = \frac{1}{\underset{y, z \in \{ \pm 1 \}^n}{\max} \rho^{\mathbb{R}} \left[ {A_c}^{-1} D_y \Delta D_z \right]}$.

## 2 kinds of warnings might print

- Jiri Rohn uses his own code to answer the optimization problem: Max $c^T \cdot x$ subject to $A \cdot x \leq b$ - which is needed during the process of checking regularity or singularity of the interval matrix of interest ($\star$) - while we use `scipy.optimize.linprog(-c, A_ub=A, b_ub=b, method="simplex").x` which solves: $\min -c^T \cdot x$ subject to $A \cdot x \leq b$. However the "simplex" method in `scipy.optimize.linprog` may not converge, raising a ValueError in `regsuffcondqz`.
- REGularIty/SINGularity of interval matrix program may be stopped after reaching prescribed number of iterations, raising a RuntimeError in `regising`.