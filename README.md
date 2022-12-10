# Jiri Rohn's algorithm to check P-property of a $(N \times N)$ matrix using **REG**ular**I**ty/**SING**ularity of interval matrix
### J. Rohn wrote his algorithm on matlab; here is a traduction of his algorithm in python.

## How to use
- Open Terminal
- Change the current working directory to the location where you want to clone this repository
- Type `git clone https://github.com/[...]` and press enter
- Go to the location where you cloned the repository
- Open check_p_property.ipynb

## The algorithm

The work presented on check_p_property.ipynb illustrates the phase transition for the P-matrix property of $I - \frac{X_N}{\alpha \sqrt{N}}$ according to the value of $\alpha$. 

- the matrix $I$ is the $(N \times N)$ identity matrix
- the matrix $X_N$ is a $(N \times N)$ random matrix with centeredentries of variance 1.
- $\alpha$ is a parameter on which depend the P-property of the matrix

This investigation is useful in theoretical ecology especially. To investigate the phase transition, we check the P-property of the matrix $I - \frac{X_N}{\alpha \sqrt{N}}$ for 15 values of $\alpha$ spread around the theoretical value of the phase change. 
- We use $\alpha \in$ `np.linspace(0.001,2,15)` for non-hermitian matrix for which the phenomenon of behavior change is supposed to occur at $\alpha = 1$. 
- We use $\alpha \in$ `np.linspace(1,3,15)` for hermitian matrix for which the phenomenon of behavior change is supposed to occur at $\alpha = 2$.

We then produce a Montecarlo estimate (for each value of $\alpha$) of the probability of being a P-matrix or not based on 50 computations. The value $\frac{1}{\alpha}$ is called *interaction strength*.

The P-matrix problem is NP-hard [Theorem 3.4](https://doi.org/10.1137/0617062), however Jiri Rohn proposed an algorithm which might converge quickly in some favourable cases, or alternatively which explore exhaustively the principal minors. Rohn's algorithm is supposed to run in polynomial time $\mathcal O (N^3)$ in convenient cases.

## Heuristic proof of Jiri Rohn's algorithm

```python 
def pmatrix(A):
    n = A.shape[0]
    I = np.eye(n)
    if np.linalg.matrix_rank(A - I) == n:
        Ac = np.linalg.inv(A - I) @ (A + I)
        S = regising(Ac, I)
    return (S is empty)
```

This algorithm is based on [Jiri Rohn's theorem 2](http://dx.doi.org/10.1007/s11590-011-0318-y). The `regising`($A_c$, $\Delta$) program considers an exhaustive list of methods to determine the **REG**ular**I**ty or the **SING**ularity of the interval $[A_c - \Delta, A_c + \Delta]$. [Jiri Rohn's theorem 2](http://dx.doi.org/10.1007/s11590-011-0318-y) states that for $(A - I)$ non-singular, $A$ is a P-Matrix $\iff$ $[ (A-I)^{-1} (A + I) - I, (A-I)^{-1} (A + I) + I ]$ is regular, i.e. it does not contain any singular matrix $S$. This interval matrix takes the form $[A_c - \Delta,A_c + \Delta]$ with $A_c = (A-I)^{-1} (A + I)$ and $\Delta = I$.

`regising` checks the regularity/singularity of the interval matrix by returning a matrix $S$ singular, if one has been found in the interval matrix (i.e. singular interval) or a value $S = [ \ ]$ empty if no singular matrix has been found in the interval (i.e. regular interval). It investigates the following methods:

**Conditions for the existence of a singular matrix**

- *midpoint matrix $A_c$*:
    the midpoint matrix $A_c$ of the interval matrix $[A_c \pm \Delta]$ is singular.
    
- *diagonal condition* [Theorem 2.1 from Oettli and Prager](https://doi.org/10.1137/S0895479896310743):
    $|A_c x| \leq \Delta |x|$ has a non trivial (i.e. non zero) solution $x$.
    
- *steepest determinant descent* [Algorithm 5.1](https://doi.org/10.1016/0024-3795(89)90004-9):
    investigate determinant bounds of the interval matrix (i.e. the hull of matrices determinant for matrices in the interval).
    
- *two Qz-matrices* [Theorem 4.3](https://doi.org/10.1137/S0895479896313978):
    the linear programming problem ($\star$) maximize $z^T x$ subject to $(A_c - \Delta \cdot diag(z)) x \leq 0$ and $diag(z) \cdot x \geq 0$, is unbounded for some $z \in \{ \pm 1 \}^n$.
    
- *main algorithm* [Theorem 2.2 - find the singular matrix](https://doi.org/10.1137/0614007):
    loop on $\{ \pm 1 \}^n$ to identify the possible singular matrix which should have the specific form.
    
- *symmetrization* [Sections 4 and 5 from Rex and Rohn](https://doi.org/10.1137/S0895479896310743):
    both of the following conditions imply the singularity of $[A_c \pm \Delta]$:
    - $\lambda_{\max}({A_c}^T A_c) \leq \lambda_{\min}(\Delta^T \Delta)$
    - $\Delta^T \Delta - {A_c}^T A_c$ positive definite

**Conditions for the regularity of the interval**

- *Beeck's condition* [Corollary 3.2 from Beeck](https://doi.org/10.1137/S0895479896310743):
    $\rho (|{A_c}^{-1}| \Delta) < 1$ is regular (for $A_c$ non singular).

- *symmetrization* [Sections 4 and 5 from Rex and Rohn](https://doi.org/10.1137/S0895479896310743):
    both of the following conditions imply the regularity of $[A_c \pm \Delta]$:
    - $\lambda_{\max}(\Delta^T \Delta) < \lambda_{\min}({A_c}^T A_c)$
    - ${A_c}^T A_c - | \Delta^T \Delta | I$ is positive definite

- *two Qz-matrices* [Theorem 4.3](https://doi.org/10.1137/S0895479896313978):
    the linear programming problem ($\star$) is bounded for all $z \in \{ \pm 1 \}^n$.  

- *main algorithm* [Theorem 2.2 - all matrices are non singular](https://doi.org/10.1137/0614007):
    loop on $\{ \pm 1 \}^n$ to check there is no singular matrix in the whole interval. This last track is the most expensive since in case the matrix is a P-matrix and no one of the conditions presented above succeeded to prove it, the algorithm will investigate the values of the sign real spectral radius.


## 2 kinds of warnings might print

- Jiri Rohn uses his own code to answer the optimization problem: Max $c^T \cdot x$ subject to $A \cdot x \leq b$, which is needed during the process of checking regularity or singularity of the interval matrix of interest ($\star$), while we use `scipy.optimize.linprog(-c, A_ub=A, b_ub=b, method="simplex").x` which solves: Min $-c^T \cdot x$ subject to $A \cdot x \leq b$. However the "simplex" method in `scipy.optimize.linprog` may not converge, raising a ValueError in `regsuffcondqz`.
- REGularIty/SINGularity of interval matrix program may be stopped after reaching prescribed number of iterations, raising a RuntimeError in `regising`.