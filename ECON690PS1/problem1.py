"""
Problem1.py

Streamlined code to:
1) generate data X_i ~ N(1,1), U_i|X_i ~ N(0, X_i^2), Y_i = beta0 + beta1*X_i + U_i
2) compute OLS and HC1 robust SEs, 95% CI for beta1
3) check CI coverage for beta1=1
4) repeat trials (1000) and report coverage and histogram of beta1_hat
5) repeat for n in [10,50,100,500,1000] and summarize

Outputs:
- prints answers for each question
- saves histograms and coverage_summary.csv in ECON690PS1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_data(n, beta0=1.0, beta1=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X = rng.normal(loc=1.0, scale=1.0, size=n)
    U = np.array([rng.normal(loc=0.0, scale=abs(x)) for x in X])
    Y = beta0 + beta1 * X + U
    return X, Y


def ols_hc1(X, Y):
    n = len(Y)
    Xmat = np.column_stack((np.ones(n), X))
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat)
    beta = XtX_inv @ Xmat.T @ Y
    resid = Y - Xmat @ beta
    S = np.diag(resid ** 2)
    k = Xmat.shape[1]
    meat = Xmat.T @ S @ Xmat
    scale = n / (n - k)
    var_beta = scale * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(var_beta))
    return beta, se


def one_simulation(n, seed=None, beta0=1.0, beta1=1.0):
    rng = np.random.default_rng(seed)
    X, Y = generate_data(n, beta0, beta1, rng=rng)
    beta_hat, se = ols_hc1(X, Y)
    # t critical
    try:
        from scipy import stats
        tcrit = stats.t.ppf(0.975, n - 2)
    except Exception:
        tcrit = 1.96
    lower = beta_hat[1] - tcrit * se[1]
    upper = beta_hat[1] + tcrit * se[1]
    contains = (lower <= beta1) and (beta1 <= upper)
    return float(beta_hat[0]), float(beta_hat[1]), float(se[0]), float(se[1]), lower, upper, contains


def repeated_experiment(n, trials=1000, beta0=1.0, beta1=1.0, base_seed=0, outdir='.'):
    rng = np.random.default_rng(base_seed)
    beta1_hats = np.empty(trials)
    covered = 0
    ses = np.empty(trials)
    for t in range(trials):
        seed = int(rng.integers(0, 2**31 - 1))
        _, b1, _, se1, lower, upper, contains = one_simulation(n, seed=seed, beta0=beta0, beta1=beta1)
        beta1_hats[t] = b1
        ses[t] = se1
        if contains:
            covered += 1

    # Save histogram
    plt.figure(figsize=(7,4))
    plt.hist(beta1_hats, bins=30, color='C0', alpha=0.8)
    plt.axvline(beta1, color='k', linestyle='--', label=f'True beta1={beta1}')
    plt.title(f'Histogram of beta1_hat (n={n}, trials={trials})')
    plt.xlabel('beta1_hat')
    plt.ylabel('Frequency')
    plt.legend()
    fname = os.path.join(outdir, f'beta1_hist_n{n}_t{trials}.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

    return beta1_hats, covered


def main():
    outdir = os.path.dirname(__file__)
    # 1) single generation for n=10 and show one sample
    print('Question 1: single dataset (n=10)')
    beta0, beta1 = 1.0, 1.0
    X, Y = generate_data(10, beta0=beta0, beta1=beta1, rng=np.random.default_rng(123))
    df = pd.DataFrame({'X': X, 'Y': Y})
    print(df.to_string(index=False))

    # 2) compute OLS and robust SEs and CI
    b0, b1, se0, se1, lower, upper, contains = one_simulation(10, seed=123, beta0=beta0, beta1=beta1)
    print('\nQuestion 2: OLS & robust SEs (n=10)')
    print(f'Estimated beta0 = {b0:.6f}, beta1 = {b1:.6f}')
    print(f'Robust SEs: se0 = {se0:.6f}, se1 = {se1:.6f}')
    print(f'95% CI for beta1: [{lower:.6f}, {upper:.6f}]')

    # 3) does CI contain true beta1=1?
    print('\nQuestion 3: Does CI contain beta1=1?')
    print('Contains' if contains else 'Does NOT contain')

    # 4) Repeat 1000 times for n=10
    print('\nQuestion 4: Repeat 1000 times (n=10)')
    trials = 1000
    beta1_hats_10, covered_10 = repeated_experiment(10, trials=trials, base_seed=2026, outdir=outdir)
    print(f'Coverage count: {covered_10} out of {trials} ({covered_10/trials*100:.2f}%)')
    print(f'Histogram saved to {os.path.join(outdir, f"beta1_hist_n10_t{trials}.png")}')

    # 5) Repeat for other sample sizes
    print('\nQuestion 5: Repeat for n in [10,50,100,500,1000]')
    sample_sizes = [10, 50, 100, 500, 1000]
    results = []
    for n in sample_sizes:
        print(f'Running n={n} (trials={trials})...')
        beta1_hats, covered = repeated_experiment(n, trials=trials, base_seed=12345 + n, outdir=outdir)
        pct = covered / trials * 100
        results.append({'n': n, 'covered': covered, 'pct': pct})
        print(f'n={n}: coverage {covered}/{trials} ({pct:.2f}%) - histogram: beta1_hist_n{n}_t{trials}.png')

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(outdir, 'coverage_summary.csv'), index=False)
    print('\nSaved coverage_summary.csv')
    print(summary)


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd

def generate_data(n=10, beta0=1.0, beta1=1.0, seed=None):
    rng = np.random.default_rng(seed)
    # Xi ~ N(1, 1)
    X = rng.normal(loc=1.0, scale=1.0, size=n)
    # Ui | Xi ~ N(0, Xi^2)
    U = np.array([rng.normal(loc=0.0, scale=abs(x)) for x in X])
    # Yi = beta0 + Xi * beta1 + Ui
    Y = beta0 + X * beta1 + U
    return X, U, Y

def main():
    n = 10
    X, U, Y = generate_data(n=n, seed=123)
    df = pd.DataFrame({"X": X, "U": U, "Y": Y})
    print(df.to_string(index=False))
    # also save to CSV
    df.to_csv("problem1_data.csv", index=False)

if __name__ == "__main__":
    main()
