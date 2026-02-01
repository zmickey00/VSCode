import numpy as np
from typing import Sequence, Union

def f(P: Union[Sequence[Sequence[float]], np.ndarray], q: Union[Sequence[float], np.ndarray], y: int, n: int) -> float:
    """
    Return P(X_n = y) for a discrete-time Markov chain given an initial distribution q.

    Parameters
    - P: N x N transition matrix (rows sum to 1). Accepts nested lists or numpy arrays.
    - q: initial probability distribution vector (0-indexed).
    - y: state in {1, ..., N} (1-based indexing).
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
    if not (1 <= y <= N):
        raise ValueError("y must be an integer in 1..N (1-based indexing)")

    q_vec = np.array(q, dtype=float)
    if q_vec.ndim != 1 or q_vec.shape[0] != N:
        raise ValueError(f"q must be a vector of length {N}")
    if not np.all(q_vec >= 0) or not np.isclose(np.sum(q_vec), 1.0):
        raise ValueError("q must be a valid probability distribution (non-negative, sums to 1)")

    An = np.linalg.matrix_power(A, n)
    dist_n = q_vec @ An
    return float(dist_n[y - 1])


def g(P: Union[Sequence[Sequence[float]], np.ndarray], q: Union[Sequence[float], np.ndarray], n: int, seed: int | None = None) -> list:
    """
    Simulate one path (y1, ..., yn) of the Markov chain starting from an initial distribution `q`.

    Parameters
    - P: N x N transition matrix (rows sum to 1).
    - q: initial probability distribution vector.
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
    q_vec = np.array(q, dtype=float)
    if q_vec.ndim != 1 or q_vec.shape[0] != N:
        raise ValueError(f"q must be a vector of length {N}")
    if not np.all(q_vec >= 0) or not np.isclose(np.sum(q_vec), 1.0):
        raise ValueError("q must be a valid probability distribution (non-negative, sums to 1)")

    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    
    states = np.arange(N)
    initial_state_0_indexed = rng.choice(states, p=q_vec)

    path = []
    current = initial_state_0_indexed
    for _ in range(n):
        probs = A[current]
        next_state = rng.choice(N, p=probs)
        path.append(int(next_state + 1))
        current = next_state

    return path


def approx_p(P: Union[Sequence[Sequence[float]], np.ndarray], q: Union[Sequence[float], np.ndarray], y: int, n: int, trials: int = 10000, seed: int | None = None) -> float:
    """
    Approximate P(X_n = y) by Monte Carlo given initial distribution q.
    It runs `g` `trials` times and compute fraction with X_n = y.

    If `n == 0`, returns q[y-1].
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    A = np.array(P, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("P must be a square (N x N) matrix")
    N = A.shape[0]

    if not (1 <= y <= N):
        raise ValueError("y must be an integer in 1..N (1-based indexing)")

    q_vec = np.array(q, dtype=float)
    if q_vec.ndim != 1 or q_vec.shape[0] != N:
        raise ValueError(f"q must be a vector of length {N}")

    if n == 0:
        return q_vec[y-1]

    rng = np.random.default_rng(seed)
    count = 0
    for t in range(trials):
        # use per-trial random integers from rng to avoid reseeding
        path = g(P, q, n, seed=rng.integers(0, 2**31 - 1))
        if path[-1] == y:
            count += 1

    return count / trials

