import numpy as np
from hsnf import smith_normal_form

from lineardiophantine.utils import (is_integer_matrix, scalar_product,
                                     strictly_greater)


def solve_on_n(A, b):
    """
    Solve the Diophantine system AX = b over non-negative integers (N)
    using the Contejean-Devie algorithm.

    This function finds all non-negative integer solutions to the linear system:

        A @ X = b,  where A is an (m × n) integer matrix and b is an m-dimensional integer vector.

    The solution set consists of:
    - **Unhomogeneous solutions** (particular solutions)
    - **Homogeneous solutions** (solutions to A @ X = 0)

    Parameters
    ----------
    A : numpy.ndarray (shape: m × n)
        The coefficient matrix of the system.

    b : numpy.ndarray (shape: m,)
        The right-hand side vector.

    Returns
    -------
    unhom_basis : list of list of int
        A basis for the particular solutions to A @ X = b (unhomogeneous part).

    hom_basis : list of list of int
        A basis for the homogeneous solutions to A @ X = 0.

    Notes
    -----
    - The algorithm follows the Contejean-Devie method, which iteratively constructs the solution
      space by generating integer vectors satisfying the system while pruning redundant candidates.
    - The function enforces non-negativity constraints.
    - The resulting solution basis can be used to express all solutions as:

        X = X_p + Σ (λ_i * X_h_i),  where X_p is a particular solution,
        X_h_i are homogeneous solutions, and λ_i are non-negative integers.

    References
    ----------
    * Contejean, E., & Devie, H. (1994). An Efficient Incremental Algorithm for Solving
      Systems of Linear Diophantine Equations. Information and Computation, 113(1), 143-172.
    """

    # Initial Setup
    A = np.insert(A, 0, (-b).reshape(1, -1), axis=1)
    n_cols = A.shape[1]
    Id = np.identity(n_cols, dtype=int)

    P = {tuple(Id[:, i]) for i in range(n_cols)}
    B = set()

    # Precompute row-wise dot products for efficiency
    A_dot_Id = {tuple(Id[:, i]): A @ Id[:, i] for i in range(n_cols)}

    # Main Algorithm
    while P:
        P_new = set()
        for x in P:
            x_np = np.array(x)

            if tuple(A @ x_np) == (0,) * A.shape[0]:  # Check if x is a solution
                B.add(x)
            else:
                for b_vec in B:
                    if strictly_greater(x, b_vec):  # Check dominance condition
                        break
                else:
                    for i in range(n_cols):
                        if i > 0 or (i == 0 and x[0] < 0):  # Constraint on x[0]
                            Ax = tuple(A @ x_np)
                            Axi = A_dot_Id[tuple(Id[:, i])]
                            if scalar_product(Ax, Axi) < 0:
                                v = tuple(x_np + Id[:, i])
                                P_new.add(v)

        P = P_new

    # Extract Basis Solutions
    unhom_basis = [list(l[1:]) for l in B if l[0] == 1]
    hom_basis = [list(l[1:]) for l in B if l[0] == 0]

    return unhom_basis, hom_basis


def solve_on_z(A, b):
    """
    Solve the Diophantine system A * X = b over the integers using the Smith normal form method.

    Parameters
    ----------
    A : np.ndarray
        A two-dimensional NumPy array of shape (m, n) representing the coefficient matrix.
    b : np.ndarray
        A two-dimensional NumPy array of shape (m, 1) representing the right-hand side vector.

    Returns
    -------
    unhom_basis : list of list of int
        A basis for the particular solutions to A @ X = b (unhomogeneous part).

    hom_basis : list of list of int
        A basis for the homogeneous solutions to A @ X = 0.
    """

    # Compute Smith normal form
    SNF, U, V = smith_normal_form(A)
    b = b.reshape(-1, 1)
    c = np.dot(U, b)

    r = 0
    while r < min(SNF.shape[0], SNF.shape[1]) and SNF[r, r] != 0:
        r = r + 1

    D = SNF[:r, :r]
    c1 = c[:r, :]
    c2 = c[r:, :]

    Dinv = np.linalg.inv(D)

    if sum(c2) == 0 and is_integer_matrix(np.dot(Dinv, c1)):
        # Extract Basis Solutions

        sol = np.dot(Dinv, c1)
        if r < A.shape[1]:
            sol = np.vstack([sol, np.array([[0]] * (A.shape[1] - r))])
        sol = np.dot(V, sol)
        sol = sol.astype(int)
        unhom_basis = (sol.T).tolist()
        NS = V[:, -(A.shape[1] - r) :]
        NS = NS.astype(int)
        hom_basis = (NS.T).tolist()

        return unhom_basis, hom_basis
    else:
        return [], []
