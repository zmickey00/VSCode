# Markov n-step transition probability

This small module provides a function `f(P, x, y, n)` that returns the n-step transition probability
P(X_n = y | X_0 = x) for a Markov chain with transition matrix `P`.

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the example:

```bash
python example.py
```

Notes
- `P` may be a nested list or a NumPy array.
- `x` and `y` use 1-based indexing (states 1..N).
- `n` must be a non-negative integer.
