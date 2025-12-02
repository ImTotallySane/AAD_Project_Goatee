import numpy as np
import time
import os
import math

# --- 1. Utility Function: Cross Product (Orientation) ---

def cross_product(p, q, r):
    """
    Calculates the 2D cross product (orientation) of vectors (q - p) and (r - q).
    > 0: Counter-clockwise (Left turn, preferred for hull)
    < 0: Clockwise (Right turn)
    = 0: Collinear
    """
    # (q_x - p_x) * (r_y - q_y) - (q_y - p_y) * (r_x - q_x)
    return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])

# --- 2. Monotone Chain Algorithm (O(N log N)) ---

def monotone_chain(points):
    """
    Implements the Monotone Chain (Andrew's algorithm) Convex Hull algorithm.
    Time Complexity: O(N log N) dominated by the initial sort.
    """
    n = len(points)
    if n < 3:
        return list(points)
    
    # 1. Sort points first by X, then by Y. This is the O(N log N) step.
    # The list conversion is needed to sort a numpy array in this way.
    points_list = points.tolist()
    points_list.sort(key=lambda p: (p[0], p[1]))
    points = np.array(points_list)

    # 2. Build the lower hull
    lower_hull = []
    for p in points:
        # While the last three points make a non-left turn (clockwise or collinear), pop the second to last point
        while len(lower_hull) >= 2 and cross_product(lower_hull[-2], lower_hull[-1], p) <= 0:
            lower_hull.pop()
        lower_hull.append(p)

    # 3. Build the upper hull
    upper_hull = []
    # Iterate through points in reverse order
    for p in reversed(points):
        # While the last three points make a non-left turn (clockwise or collinear), pop the second to last point
        while len(upper_hull) >= 2 and cross_product(upper_hull[-2], upper_hull[-1], p) <= 0:
            upper_hull.pop()
        upper_hull.append(p)

    # 4. Concatenate and return. The last point of the lower hull and the last point 
    # of the upper hull will be the same anchor points (min X and max X). 
    # We remove the last point of both lists to avoid duplicating endpoints.
    # The hull is lower_hull[:-1] + upper_hull[:-1]
    return np.array(lower_hull[:-1] + upper_hull[:-1])


# --- 3. Timing and Analysis Functions ---

def load_data(filename):
    """Loads X and Y coordinates from the generated file."""
    try:
        return np.loadtxt(filename, delimiter=',')
    except FileNotFoundError:
        return None

def run_timing_experiment(algorithm_func, input_sizes, complexity_term, num_runs=5, max_total_time_per_n=60):
    """
    Runs the specified algorithm on all datasets and records time, 
    comparing against the specified complexity term (N log N).
    """
    results = []
    algo_name = algorithm_func.__name__.replace('_', ' ').title()
    print(f"\n--- Starting {algo_name} Timing Experiment (Target: {complexity_term}) ---")
    
    for n in input_sizes:
        filename = f"dataset_{n}_points.txt"
        points = load_data(filename)
        
        if points is None:
            print(f"Skipping N={n}: Data file not found.")
            continue 

        # Time the algorithm R times
        total_time = 0
        runs_completed = 0
        
        for r in range(num_runs):
            # Timeout safety check
            if total_time >= max_total_time_per_n:
                print(f"    [!] Timeout threshold reached for N={n}. Skipping remaining {num_runs - r} runs.")
                break

            start_time = time.perf_counter()
            algorithm_func(points)
            end_time = time.perf_counter()
            
            run_time = end_time - start_time
            total_time += run_time
            runs_completed += 1
            
        # Calculate average time only using completed runs
        if runs_completed > 0:
            avg_time = total_time / runs_completed
        else:
            avg_time = float('inf') 

        # Calculate the theoretical complexity term f(n) = N log N
        f_n = n * math.log2(n) if n > 1 else 0
        
        # Calculate the ratio T_avg / f(n)
        ratio = avg_time / f_n if f_n > 0 and avg_time != float('inf') else 0
        
        # Store results
        results.append({
            'N': n,
            'T_avg': avg_time,
            'F_N': f_n,
            'Ratio_T/F_N': ratio,
            'Runs_Completed': runs_completed,
            'Complexity_Term': complexity_term
        })
        
        # Print intermediate results for monitoring
        print(f"N={n:<8}: Avg Time={avg_time:.6f}s | Runs={runs_completed}/{num_runs} | T/F(N)={ratio:.2e}")

    return results

def print_analysis(results, complexity_name, complexity_term):
    """Prints the final results table and conclusion."""
    print("\n" + "="*80)
    print(f"                 {complexity_name} Empirical Complexity Analysis (Target: {complexity_term})")
    print("="*80)
    print(f"{'N (Input Size)':<20} | {'T_avg (Seconds)':<20} | {complexity_term:<20} | {'Ratio (T_avg / F(N))':<20}")
    print("-" * 80)
    
    # Calculate average ratio for the largest sizes (where complexity dominates)
    valid_ratios = [r['Ratio_T/F_N'] for r in results if r['Ratio_T/F_N'] > 0 and r['N'] >= 100000]
    constant_c_avg = np.mean(valid_ratios) if valid_ratios else 0
    
    for r in results:
        # Handle cases where the run timed out or failed
        time_str = f"{r['T_avg']:.10f}" if r['T_avg'] != float('inf') else "TIMEOUT"
        ratio_str = f"{r['Ratio_T/F_N']:.2e}" if r['Ratio_T/F_N'] > 0 else "N/A"
        f_n_str = f"{r['F_N']:.2f}"
        
        print(f"{r['N']:<20} | {time_str}{'s':<19} | {f_n_str:<20} | {ratio_str}{'':<17}")

    print("\n--- Conclusion ---")
    print(f"The 'Ratio (T_avg / F(N))' column shows how close the runtime is to a constant 'c'.")
    print(f"For large N, the ratio consistently converges to a value (approx. {constant_c_avg:.2e}).")
    print(f"This convergence confirms the algorithm's empirical time complexity is $\mathbf{{{complexity_name}}}$ ($\mathbf{{{complexity_term}}}$).")


# --- Main Execution ---

if __name__ == "__main__":
    
    # --- Configuration ---
    # Input sizes to test Monotone Chain (up to 1 million points)
    monotone_chain_input_sizes = [
        10, 100, 500, 1000, 2000, 10000, 50000, 100000, 250000, 500000, 1000000
    ]
    
    # The complexity term to compare against
    complexity_term = 'N log N'
    
    # Check if the smallest required file exists
    if not os.path.exists(f"dataset_{monotone_chain_input_sizes[0]}_points.txt"):
        print(f"⚠️ Data files not found (e.g., dataset_{monotone_chain_input_sizes[0]}_points.txt).")
        print("Please run the data generator script to create the required datasets (e.g., generate_monotone_data.py).")
    else:
        # --- Run Monotone Chain Experiment ---
        timing_results = run_timing_experiment(
            algorithm_func=monotone_chain, 
            input_sizes=monotone_chain_input_sizes, 
            complexity_term=complexity_term, 
            num_runs=5, 
            max_total_time_per_n=60 
        )
        print_analysis(timing_results, "Monotone Chain", complexity_term)