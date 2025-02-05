import numpy as np
from lineardiophantine.solve import solve_on_n, solve_on_z


def test_solve_on_n():
    A = np.array([[-1, 1, 2, -3], [-1, 3, -2, -1]])
    b = np.array([1, 1])
    result_unhom_basis, result_hom_basis = solve_on_n(A, b)
    expected_unhom_basis = [[3, 2, 1, 0]]
    expected_hom_basis = [[4, 2, 1, 0], [0, 1, 1, 1]]
    assert result_unhom_basis == expected_unhom_basis
    assert result_hom_basis == expected_hom_basis


def test_solve_on_z():
    A = np.array([[-1, 1, 2, -3], [-1, 3, -2, -1]])
    b = np.array([1, 1])
    result_unhom_basis, result_hom_basis = solve_on_z(A, b)
    expected_unhom_basis = [[-1, 0, 0, 0]]
    expected_hom_basis = [[4, 2, 1, 0], [-4, -1, 0, 1]]
    assert result_unhom_basis == expected_unhom_basis
    assert result_hom_basis == expected_hom_basis
    A = np.array([[-1, 1, 2, -3], [-1, 3, -2, -1]])
    b = np.array([-1, 0])
    result_unhom_basis, result_hom_basis = solve_on_z(A, b)
    expected_unhom_basis = []
    expected_hom_basis = []
    assert result_unhom_basis == expected_unhom_basis
    assert result_hom_basis == expected_hom_basis
