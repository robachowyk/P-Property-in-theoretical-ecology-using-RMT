import numpy as np
from matplotlib import pyplot as plt

def is_singular(matrix):
    assert matrix.shape[0] == matrix.shape[1]
    if np.linalg.det(matrix) == 0:
        return True
    else:
        return False

def is_sym_pos_def(A):
    return np.allclose(A, A.T, rtol=1e-05, atol=1e-08, equal_nan=False) & np.all(np.linalg.eigvals(A) > 0)

def generate_p_matrix(n):
    """
    n : square matrix dimension
    references : (PAGE 5) https://invenio.nusl.cz/record/81055/files/content.csg.pdf
                 (THEOREM 2) http://uivtx.cs.cas.cz/~rohn/publist/genpmat.pdf
    """
    C = 2*np.random.uniform(size=(n,n))-1
    C_inv = np.linalg.inv(C)
    D = np.random.uniform(size=(n,n))
    alpha = 0.95 / spectral_radius(np.abs(C_inv) @ D)
    return np.linalg.inv(C - alpha*D) @ (C + alpha*D)

def singular_eigvals(A):
    u, s, vh = np.linalg.svd(A)
    return s

def real_spectral_radius(A):
    real_eigs = np.linalg.eigvals(A)[ np.isreal(np.linalg.eigvals(A)) ]
    if len(real_eigs)==0:
        return 0
    else:
        return np.max( np.abs( np.linalg.eigvals(A)[ np.isreal(np.linalg.eigvals(A)) ] ) )
    
def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))

def symmetrize(a):
    return a + a.T - np.diag(np.diagonal(a))

def get_wigner(loc, scale, size):
    a = np.random.normal(loc=loc, scale=scale, size=size)
    a = np.round_(a, 3)
    a = np.triu(a)
    return symmetrize(a)

def plot_spectrum(B, radius, grid, title, color_dots, color_circle):
    eig_B = np.linalg.eigvals(B)
    t = np.linspace(0, 2*np.pi, 100)
    fig = plt.figure(figsize=(12,7))
    if grid:
        plt.grid()
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")
    plt.title(title)
    plt.plot(eig_.real, eig_.imag, '.', color = color_dots)
    plt.plot(radius*np.cos(t), radius*np.sin(t), color = color_circle)
    plt.plot(0, 0, marker = "+", color = color_circle)
    plt.axis("scaled")
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.show()
    plt.close()

def block_matrix_normal(n_size, beta, mu, sigma):
    """
    Creation of the block matrix associated
    Parameters
    ----------
    n_size : int,
        dimension of the matrix.
    beta : list (matrix),
        correspond to the size of each block.
    mu : list (matrix),
        correspond to the mean of each block.
    sigma : list (matrix)
        correspond to the variance of each block.
    Returns
    -------
    A : list (matrix)
        The filled block matrix.

    N_SIZE = 1000
    beta = [1/2,1/2]
    sigma = np.array([[1,1/4],[1/4,1]])/np.sqrt(N_SIZE)

    mu = np.array([[0,4],[4,0]])/N_SIZE

    BMN = block_matrix_normal(1000,beta,mu,sigma)
    """
    B = np.size(beta)
    A = np.random.randn(
        int(n_size*beta[0]), int(n_size*beta[0]))*sigma[0, 0]+mu[0, 0]
    for i in range(1, B):
        A_bis = np.random.randn(
            int(n_size*beta[0]), int(n_size*beta[i]))*sigma[0, i]+mu[0, i]
        A = np.concatenate([A, A_bis], axis=1)
    for j in range(1, B):
        Aj = np.random.randn(
            int(n_size*beta[j]), int(n_size*beta[0]))*sigma[j, 0]+mu[j, 0]
        for k in range(1, B):
            A_bisj = np.random.randn(
                int(n_size*beta[j]), int(n_size*beta[k]))*sigma[j, k]+mu[j, k]
            Aj = np.concatenate([Aj, A_bisj], axis=1)
        A = np.concatenate([A, Aj], axis=0)
    return A