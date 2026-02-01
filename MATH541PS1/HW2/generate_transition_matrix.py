
import numpy as np
import re
import requests

def download_war_and_peace(url="https://www.gutenberg.org/files/2600/2600-0.txt"):
    """
    Downloads the text of "War and Peace" from Project Gutenberg.
    """
    response = requests.get(url)
    response.encoding = 'utf-8'
    return response.text

def create_transition_matrix(text):
    """
    Creates a transition matrix from a given text.
    """
    # Clean the text
    text = text.lower()
    text = re.sub(r'[^a-z]', '', text) # Only keep letters
    
    # Initialize a 26x26 matrix with ones for Laplace smoothing
    transition_counts = np.ones((26, 26))
    
    # Create a mapping from character to index (a=0, b=1, ...)
    char_to_int = {chr(ord('a') + i): i for i in range(26)}
    
    # Iterate through the text to count transitions
    for i in range(len(text) - 1):
        current_char = text[i]
        next_char = text[i+1]
        
        if current_char in char_to_int and next_char in char_to_int:
            u = char_to_int[current_char]
            v = char_to_int[next_char]
            transition_counts[u, v] += 1
            
    # Normalize the counts to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = transition_counts / row_sums
    
    return transition_matrix

def save_matrix_to_csv(matrix, filename="war_and_peace_transition_matrix_letters_only.csv"):
    """
    Saves a matrix to a CSV file.
    """
    np.savetxt(filename, matrix, delimiter=',')

if __name__ == "__main__":
    print("Downloading 'War and Peace'...")
    war_and_peace_text = download_war_and_peace()
    
    print("Creating the transition matrix...")
    transition_matrix = create_transition_matrix(war_and_peace_text)
    
    print("Saving the transition matrix to CSV...")
    save_matrix_to_csv(transition_matrix, "VSCode/MATH541PS1/HW2/war_and_peace_transition_matrix_letters_only.csv")
    
    print("Done.")
