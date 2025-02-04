import numpy as np

from lineardiophantine.utils import (is_integer_matrix, scalar_product,
                                     strictly_greater)


def test_is_integer_matrix():
    A = np.array([[-8, 0, 2], [-5, -3, 12]])
    result = is_integer_matrix(A)
    expected = True
    assert result == expected
    A = np.array([[0, 1], [2, 0.5]])
    result = is_integer_matrix(A)
    expected = False
    assert result == expected


def test_scalar_product():
    v1 = np.array([-1, -2, 10])
    v2 = np.array([6, 0, 1])
    result = scalar_product(v1, v2)
    expected = 4
    assert result == expected
    v1 = [-2, 1, 3]
    v2 = [4, -9, 8]
    result = scalar_product(v1, v2)
    expected = 7
    assert result == expected


def test_strictly_greater():
    v1 = [-1, -2, 10]
    v2 = [6, 0, 1]
    result = strictly_greater(v1, v2)
    expected = False
    assert result == expected
    v1 = [4, 9, 8]
    v2 = [-2, 1, 3]
    result = strictly_greater(v1, v2)
    expected = True
    assert result == expected
    v1 = [4, 9, 8]
    result = strictly_greater(v1, v1)
    expected = False
    assert result == expected
