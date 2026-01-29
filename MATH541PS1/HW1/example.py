from markov import f
import numpy as np

def main():
    # Simple 2-state Markov chain
    P = [[0.5, 0.5], [0.2, 0.8]]
    x = 1
    y = 2
    n = 3

    prob = f(P, x, y, n)
    print(f"P(X_{n}={y} | X_0={x}) = {prob:.6f}")

    # verify via direct matrix power
    A = np.array(P)
    print("Check (matrix_power):", np.linalg.matrix_power(A, n)[x - 1, y - 1])
    
    # Monte Carlo approximation using simulator g
    from markov import approx_p, g

    trials = 20000
    est = approx_p(P, x, y, n, trials=trials, seed=42)
    print(f"Monte Carlo estimate (trials={trials}): {est:.6f}")

    # show one sample path
    sample = g(P, x, n, seed=123)
    print("Sample path:", sample)

if __name__ == "__main__":
    main()
