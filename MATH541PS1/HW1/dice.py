import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def roll_dice(num_rolls=100):
    """
    Simulates rolling a standard six-sided die a specified number of times.

    Args:
        num_rolls (int): The number of times to roll the die.

    Returns:
        list: A list of the outcomes of the dice rolls.
    """
    return np.random.randint(1, 7, size=num_rolls)

def plot_dice_rolls(rolls):
    """
    Plots a bar chart of the outcomes of the dice rolls.

    Args:
        rolls (list): A list of the outcomes of the dice rolls.
    """
    counts = Counter(rolls)
    outcomes = sorted(counts.keys())
    frequencies = [counts[outcome] for outcome in outcomes]

    plt.bar(outcomes, frequencies, tick_label=outcomes)
    plt.xlabel("Dice Outcome")
    plt.ylabel("Frequency")
    plt.title("Frequency of Outcomes for 100 Dice Rolls")
    plt.xticks(range(1, 7))
    plt.grid(axis='y', alpha=0.75)
    plt.show()
if __name__ == '__main__':
    rolls = roll_dice()
    plot_dice_rolls(rolls)
