# **Solving linear Diophantine systems over natural numbers $\mathbb{N}$**

This repository contains code for solving linear Diophantine systems over natural numbers $\mathbb{N}$. We briefly recall definitions and theoretical results about linear Diophantine systems.

## **Definition**

A linear Diophantine system is a system of equations of the form 
$$\left\{
\begin{aligned}
 a_{1,1}x_1 + ... + a_{1,n}x_n & = b_1\\
 \vdots & \\
 a_{m,1}x_1 + ... + a_{m,n}x_n & = b_m\\
\end{aligned} \right. \quad (E)$$
where $a_{i,j}, b_{k} \in \mathbb{Z}$ and where we are looking for natural number solutions $(x_1,\ldots,x_n) \in \mathbb{N}^n$.

Solving the system $(E)$ is equivalent to solve the matrix equation $AX = b$ where $A = \begin{pmatrix}
a_{1,1} & \ldots & a_{1,n} \\
& \vdots & \\
a_{m,1} & \ldots & a_{m,n}
\end{pmatrix}$, $b = \begin{pmatrix}
b_1\\
\vdots\\
b_m
\end{pmatrix}$ and $X = \begin{pmatrix}
x_1\\
\vdots\\
x_n
\end{pmatrix}$.

The linear Diophantine system $AX = 0 ~ (E_h)$ is called the homogeneous system associated to $(E)$.

## **Solutions**

There exist a set of particular solutions $X_{p,1},\ldots, X_{p,r}\in \mathbb{N}^n$ of $(E)$ and a set of homogeneous solutions $X_{h,1}, \ldots, X_{h,s}\in \mathbb{N}^n$ of $(E_h)$ such that every solution $X \in \mathbb{N}^n$ of $(E)$ is of the form
$$X = X_{p,j} + \sum\limits_{i=1}^s \lambda_i X_{h,i}$$
for a $j \in ⟦1,r⟧$ and where $\lambda_i \in \mathbb{N} ~ \forall i \in ⟦1,s⟧$.

## **Solving function**

The ```solve_on_n``` function takes as parameters the matrix ```A``` and the vector ```b``` of the linear Diophantine system $(E)$. The required formats are the following:
- the matrix ```A``` has to be in format ```numpy.ndarray``` and of shape: ```(m × n)```;
- the vector ```b``` has to be in format ```numpy.ndarray``` and of shape: ```(m,)```.

The function ouputs a tuple ```(unhom_basis, hom_basis)```, where:

- ```unhom_basis``` is a list of vectors in list format corresponding to the particular solutions $X_{p,1},\ldots, X_{p,r}$ of $(E)$;
- ```hom_basis``` is a list of vectors in list format corresponding to the homogeneous solutions $X_{h,1}, \ldots, X_{h,s}$ of $(E_h)$.

## **Solving algorithm**

The ```solve_on_n``` function uses the Contejean-Devie algorithm. See

- Contejean, E., & Devie, H. (1994). An efficient incremental algorithm for solving systems of linear diophantine equations. Information and computation, 113(1), 143-172.

## **Bonus: solving over $\mathbb{Z}$**

Linear Diophantine systems are easier to solve over positive and negative integers $\mathbb{Z}$ since the solution set is a $\mathbb{Z}$-module and remark that only one particular solution is necessary to express any solution in that case. For sake of completeness we have also included a ```solve_on_z``` that solves linear Diophantine systems over positive and negative integers $\mathbb{Z}$ using the Smith normal form method.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

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
* [example_usage.ipynb](lineardiophantine/examples/example_usage.ipynb) is a notebook providing solving examples.
