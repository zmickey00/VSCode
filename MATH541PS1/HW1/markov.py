import numpy as np
from typing import Sequence, Union

def f(P: Union[Sequence[Sequence[float]], np.ndarray], x: int, y: int, n: int) -> float:
    """
    Return pn(x, y) = P(X_n = y | X_0 = x) for a discrete-time Markov chain.

    Parameters
    - P: N x N transition matrix (rows sum to 1). Accepts nested lists or numpy arrays.
    - x, y: states in {1, ..., N} (1-based indexing).
    - n: non-negative integer time step.

    Returns
    - probability as a float
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    A = np.array(P, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("P must be a square (N x N) matrix")

    N = A.shape[0]
    if not (1 <= x <= N and 1 <= y <= N):
        raise ValueError("x and y must be integers in 1..N (1-based indexing)")

    An = np.linalg.matrix_power(A, n)
    return float(An[x - 1, y - 1])


def g(P: Union[Sequence[Sequence[float]], np.ndarray], x: int, n: int, seed: int | None = None) -> list:
    """
    Simulate one path (y1, ..., yn) of the Markov chain starting from state `x`.

    Parameters
    - P: N x N transition matrix (rows sum to 1).
    - x: initial state in 1..N (1-based indexing).
    - n: number of steps to simulate (non-negative integer).
    - seed: optional RNG seed for reproducibility.

    Returns
    - list of length `n` with states in 1..N (1-based) representing X1..Xn.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    A = np.array(P, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("P must be a square (N x N) matrix")

    N = A.shape[0]
    if not (1 <= x <= N):
        raise ValueError("x must be an integer in 1..N (1-based indexing)")

    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    path = []
    current = x - 1
    for _ in range(n):
        probs = A[current]
        next_state = rng.choice(N, p=probs)
        path.append(int(next_state + 1))
        current = next_state

    return path


def approx_p(P: Union[Sequence[Sequence[float]], np.ndarray], x: int, y: int, n: int, trials: int = 10000, seed: int | None = None) -> float:
    """
    Approximate pn(x,y) by Monte Carlo: run `g` `trials` times and compute fraction with X_n = y.

    If `n == 0`, returns 1.0 if `x == y` else 0.0.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    if n == 0:
        return 1.0 if x == y else 0.0

    rng = np.random.default_rng(seed)
    count = 0
    for t in range(trials):
        # use per-trial random integers from rng to avoid reseeding
        path = g(P, x, n, seed=rng.integers(0, 2**31 - 1))
        if path[-1] == y:
            count += 1

    return count / trials

