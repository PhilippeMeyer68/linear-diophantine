# **Solving linear Diophantine systems over natural numbers**

This repository contains a code and especially a ```solve_on_n``` function for solving linear Diophantine systems over natural numbers using the Contejean-Devie algorithm.

## Solving algorithm

See

- Contejean, E., & Devie, H. (1994). An efficient incremental algorithm for solving systems of linear diophantine equations. Information and computation, 113(1), 143-172.

## **Bonus: solving over positive and negative integers**

For sake of completeness we have also included a ```solve_on_z``` that solves linear Diophantine systems over positive and negative integers using the Smith normal form method.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install lineardiophantine.

```bash
pip install lineardiophantine
```

## Usage

```python
import numpy as np

from lineardiophantine.solve import solve_on_n

# matrices of the system
A = np.array([[5, -9, 8, 6], [7, -7, 6, 9], [17, -25, 22, 21]])
b = np.array([1, 7, 9])

# returns a tuple composed of the particular and the homogeneous solutions
solve_on_n(A, b)
```

## Content description

* [solve.py](lineardiophantine/solve.py) is the python script containing the solving functions.
* [utils.py](lineardiophantine/utils.py) is a python script containing the utility functions.
* [example_usage.ipynb](lineardiophantine/examples/example_usage.ipynb) is a notebook providing theoretical explanations and practical solving examples.
