import numpy as np


def strictly_greater(v1, v2):
    """
    Check if each corresponding element in v1 is greater than or equal to v2,
    with at least one element being strictly greater.

    Parameters
    ----------
    v1 : iterable
        The first vector (e.g., list or tuple of numbers).
    v2 : iterable)
        The second vector, of the same length as v1.

    Returns
    -------
    bool
        True if v1 is strictly greater than v2 component-wise,
        meaning all elements of v1 are greater than or equal to
        the corresponding elements of v2, and at least one element
        is strictly greater. Otherwise, returns False.
    """

    return all(x >= y for x, y in zip(v1, v2)) and v1 != v2


def scalar_product(v1, v2):
    """
    Compute the scalar (dot) product of two vectors.

    Parameters
    ----------
    v1 : iterable or np.ndarray
        The first vector, either a list, tuple, or NumPy array.
    v2 : iterable or np.ndarray
        The second vector, of the same length as v1.

    Returns
    -------
    float or int:
        The scalar product of v1 and v2.
    """

    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        return np.dot(v1, v2)
    return sum(x * y for x, y in zip(v1, v2))
