import numpy as np
import random
import math
import re
import multiprocessing

# The encrypted message
encrypted_message = "col hlyg fms wit fh gtww pommf mtzty poosqet jfc pommf mtzty ryc col efvt pommf mtzty col stjtyg fms fyolms ylm pommf mtzty soam col wtg pommf mtzty In col pizt pommf mtzty jtt go qwims goo colyt et fjv col ik fms ig nwfc pommf atyt fms pfet ght vmoa at om poimp qttm ahfgj vmoa qogh at imjist ig jfc go jhe goo colyt qlg frhimp qttm htfygj coly womp jo koy oghty tfrh vmoam atzt Imstyjgfms col efvt poggf kttwimp ie hoa col gtww afmmf uljg tet plc oghty fmc kyoe ghij ptg aolwsmg col ok ghimvimp ie ahfg roeeigetmgj klww fh tct so jo fms ylwtj ght vmoa col wozt go jgyfmptyj mo atyt"

# Create a mapping from character to index (a=0, b=1, ...)
char_to_int = {chr(ord('a') + i): i for i in range(26)}

def get_transition_matrix(filename="VSCode/MATH541PS1/HW2/war_and_peace_transition_matrix_letters_only.csv"):
    """
    Reads the transition matrix from a CSV file.
    """
    try:
        # Adding a small epsilon for smoothing to prevent log(0)
        matrix = np.genfromtxt(filename, delimiter=',')
        if np.count_nonzero(matrix) == 0:
             # Fallback if matrix is empty or all zeros
            return np.full((26, 26), 1/26)
        matrix += 1e-12 
        matrix /= matrix.sum(axis=1, keepdims=True)
        return matrix
    except OSError:
        print(f"Error: Could not read {filename}.")
        print("Please make sure the file is in the correct directory.")
        # Return a uniform probability matrix if the file is not found
        return np.full((26, 26), 1/26)


def calculate_log_pl(s, text, M):
    """
    Calculates the log of Pl(s) for a given permutation s and text.
    """
    log_pl = 0.0
    
    # Create a decryption map from the permutation
    decryption_map = {chr(ord('a') + i): s[i] for i in range(26)}
    
    # Clean the text to treat it as one continuous string of letters
    cleaned_text = re.sub(r'[^a-zA-Z]', '', text).lower()

    for i in range(len(cleaned_text) - 1):
        current_char_encrypted = cleaned_text[i]
        next_char_encrypted = cleaned_text[i+1]
        
        # Decrypt characters
        current_char_decrypted = decryption_map.get(current_char_encrypted)
        next_char_decrypted = decryption_map.get(next_char_encrypted)

        if current_char_decrypted and next_char_decrypted:
            u = char_to_int[current_char_decrypted]
            v = char_to_int[next_char_decrypted]
            log_pl += math.log(M[u, v])
            
    return log_pl


def metropolis_hastings(text, M, beta=0.1, iterations=100000, run_id=None):
    """
    Performs the Metropolis-Hastings algorithm to find the decryption key.
    """
    # Start with a random permutation
    s = list('abcdefghijklmnopqrstuvwxyz')
    random.shuffle(s)
    s = "".join(s)
    
    # Calculate the initial log Pl
    log_pl_s = calculate_log_pl(s, text, M)
    
    for i in range(iterations):
        # Propose a new permutation by swapping two letters
        s_prime_list = list(s)
        i1, i2 = random.sample(range(26), 2)
        s_prime_list[i1], s_prime_list[i2] = s_prime_list[i2], s_prime_list[i1]
        s_prime = "".join(s_prime_list)
        
        # Calculate the log Pl for the new permutation
        log_pl_s_prime = calculate_log_pl(s_prime, text, M)
        
        # Calculate the acceptance ratio
        if log_pl_s_prime > log_pl_s:
            acceptance_ratio = 1.0
        # This check is important to prevent math range errors
        elif log_pl_s == -float('inf'):
            acceptance_ratio = 1.0 
        else:
            acceptance_ratio = math.exp(beta * (log_pl_s_prime - log_pl_s))
        
        # Accept or reject the new permutation
        if random.random() < acceptance_ratio:
            s = s_prime
            log_pl_s = log_pl_s_prime

        if (i + 1) % 25000 == 0:
            if run_id:
                print(f"Run {run_id}: Iteration {i+1}/{iterations}")
            else:
                print(f"Iteration {i+1}/{iterations}")
            
    return s

def decrypt_message(encrypted_message, key):
    """
    Decrypts a message using a given key.
    """
    decryption_map = {chr(ord('a') + i): key[i] for i in range(26)}
    decrypted_message = ""
    for char in encrypted_message:
        decrypted_char = decryption_map.get(char.lower(), char)
        if char.isupper():
            # Attempt to capitalize if the original was upper, but fall back to lower if not in map
            decrypted_message += decrypted_char.upper()
        else:
            decrypted_message += decrypted_char
            
    return decrypted_message

def worker(run_id):
    """A worker function to run one instance of the Metropolis-Hastings simulation."""
    print(f"Starting run {run_id}...")
    # Each process will read the matrix. This is fine for read-only data.
    transition_matrix = get_transition_matrix()
    final_key = metropolis_hastings(encrypted_message, transition_matrix, iterations=100000, run_id=run_id)
    final_log_pl = calculate_log_pl(final_key, encrypted_message, transition_matrix)
    print(f"Finished run {run_id} with log PL: {final_log_pl:.2f}")
    return (final_key, final_log_pl)


if __name__ == "__main__":
    # It's important to set the start method for compatibility, especially on macOS and Windows
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # set_start_method can only be called once, will raise an error if already set
        pass

    num_runs = 400
    print(f"Running Metropolis-Hastings {num_runs} times in parallel using up to {multiprocessing.cpu_count()} cores...")

    # A Pool of workers will run the simulations in parallel
    with multiprocessing.Pool() as pool:
        # map distributes the runs across the worker processes
        results = pool.map(worker, range(1, num_runs + 1))

    # Find the best key from all the parallel runs by comparing their log PL values
    best_key, best_log_pl = max(results, key=lambda item: item[1])

    print(f"\n--- Best Result ---")
    print(f"Best Key Found: {best_key}")
    print(f"Best Log PL: {best_log_pl:.2f}")
    
    decrypted_message = decrypt_message(encrypted_message, best_key)
    print(f"\nDecrypted Message: {decrypted_message}")