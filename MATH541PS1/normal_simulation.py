import numpy as np
import matplotlib.pyplot as plt

def simulate_normal_draws(num_draws=100):
    """
    Simulates a specified number of random draws from a standard normal distribution.

    Args:
        num_draws (int): The number of random draws to simulate.

    Returns:
        numpy.ndarray: An array of random draws from a standard normal distribution.
    """
    return np.random.randn(num_draws)

def plot_draws(draws):
    """
    Plots a histogram of the given draws.

    Args:
        draws (numpy.ndarray): The data to plot.
    """
    plt.hist(draws, bins='auto', density=True, alpha=0.7, rwidth=0.85)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of 100 Random Draws from a Standard Normal Distribution')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == '__main__':
    draws = simulate_normal_draws()
    plot_draws(draws)
