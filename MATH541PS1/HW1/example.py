from markov import f
import numpy as np

def main():
    # Simple 2-state Markov chain
    P = [[0.5, 0.5], [0.2, 0.8]]
    x = 1
    y = 2
    n = 3
    
    # initial state x=1 as a probability distribution
    q = np.zeros(len(P))
    q[x-1] = 1.0

    prob = f(P, q, y, n)
    print(f"P(X_{n}={y} | X_0={x}) = {prob:.6f}")

    # verify via direct matrix power
    A = np.array(P)
    print("Check (matrix_power):", np.linalg.matrix_power(A, n)[x - 1, y - 1])
    
    # Monte Carlo approximation using simulator g
    from markov import approx_p, g

    trials = 20000
    est = approx_p(P, q, y, n, trials=trials, seed=42)
    print(f"Monte Carlo estimate (trials={trials}): {est:.6f}")

    # show one sample path
    sample = g(P, q, n, seed=123)
    print("Sample path:", sample)
    
    # Example with a non-degenerate initial distribution
    print("\nExample with initial distribution q = [0.5, 0.5]")
    q_mixed = [0.5, 0.5]
    prob_mixed = f(P, q_mixed, y, n)
    print(f"P(X_{n}={y} | q_0=[0.5, 0.5]) = {prob_mixed:.6f}")
    
    # Check for mixed distribution
    # (0.5 * P(X_n=y|X_0=1)) + (0.5 * P(X_n=y|X_0=2))
    check_mixed = 0.5 * np.linalg.matrix_power(A, n)[0, y - 1] + 0.5 * np.linalg.matrix_power(A, n)[1, y - 1]
    print(f"Check: {check_mixed:.6f}")

    est_mixed = approx_p(P, q_mixed, y, n, trials=trials, seed=42)
    print(f"Monte Carlo estimate (trials={trials}): {est_mixed:.6f}")
    
    sample_mixed = g(P, q_mixed, n, seed=123)
    print("Sample path from mixed initial distribution:", sample_mixed)


if __name__ == "__main__":
    main()
