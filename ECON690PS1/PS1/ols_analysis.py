import numpy as np
import pandas as pd
import math

def ols_robust(df, y_col='Y', x_col='X'):
    n = df.shape[0]
    X = np.column_stack((np.ones(n), df[x_col].values))
    y = df[y_col].values.reshape(-1, 1)
    k = X.shape[1]

    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    y_hat = X @ beta_hat
    resid = (y - y_hat).reshape(-1)

    # HC1 robust variance: (n/(n-k)) * (XtX_inv @ X.T @ diag(resid^2) @ X @ XtX_inv)
    S = np.diag(resid ** 2)
    meat = X.T @ S @ X
    scale = n / (n - k)
    var_beta_hat = scale * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(var_beta_hat)).reshape(-1, 1)

    return beta_hat.reshape(-1), se.reshape(-1)

def ci_beta1(beta_hat, se, n, k, alpha=0.05):
    try:
        import scipy.stats as st
        df = n - k
        tcrit = st.t.ppf(1 - alpha / 2, df)
    except Exception:
        # fallback to normal approx
        tcrit = 1.96
    lower = beta_hat - tcrit * se
    upper = beta_hat + tcrit * se
    return float(lower), float(upper), float(tcrit)

def main():
    df = pd.read_csv('problem1_data.csv')
    n = df.shape[0]
    beta_hat, se = ols_robust(df)
    print('OLS estimates (beta0, beta1):', beta_hat)
    print('Robust SEs (se0, se1):', se)
    lower, upper, tcrit = ci_beta1(beta_hat[1], se[1], n, 2)
    print(f'95% CI for beta1: [{lower:.4f}, {upper:.4f}] (tcrit={tcrit:.4f})')

if __name__ == '__main__':
    main()
