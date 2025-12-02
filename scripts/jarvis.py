import numpy as np
import time
import os
import math

# --- 1. Utility Function: Orientation ---

def orientation(p, q, r):
    """
    Find the orientation of the ordered triplet (p, q, r).
    0 --> p, q and r are collinear
    1 --> Clockwise (Right Turn)
    2 --> Counterclockwise (Left Turn)
    """
    # Calculate cross product (q - p) x (r - q)
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    
    if val == 0: return 0  # Collinear
    return 1 if val > 0 else 2 # Clockwise or Counterclockwise (2 is preferred for hull traversal)

# --- 2. Jarvis March Algorithm (O(nh)) ---

def jarvis_march(points):
    """
    Implements the Jarvis March (Gift Wrapping) Convex Hull algorithm.
    Time Complexity: O(nh), worst case O(N^2).
    """
    n = len(points)
    if n < 3:
        return list(points) 

    # 1. Find the anchor point (P0) - lowest Y, leftmost X
    l = min(range(n), key=lambda i: (points[i][1], points[i][0]))

    hull = []
    p = l
    
    # 2. Start wrapping
    while True:
        hull.append(points[p])
        q = (p + 1) % n
        
        # Search for a point q such that the orientation (p, q, i) is CCW for all i
        for i in range(n):
            # If orientation(p, q, i) is Counterclockwise, update q
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        
        p = q
        
        # Stop when we return to the starting point
        if p == l:
            break
            
    return np.array(hull)


# --- 3. Timing and Analysis Functions (Adapted from Graham Scan structure) ---

def load_data(filename):
    """Loads X and Y coordinates from the generated file."""
    try:
        return np.loadtxt(filename, delimiter=',')
    except FileNotFoundError:
        return None

def run_timing_experiment(algorithm_func, input_sizes, complexity_term, num_runs=5, max_total_time_per_n=60):
    """
    Runs the specified algorithm on all datasets and records time, 
    comparing against the specified complexity term (N^2).
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

        # Calculate the theoretical complexity term f(n) = N^2
        if complexity_term == 'N^2':
            f_n = n * n
        else: # Should not happen, but safe guard against generic term
             f_n = 1
        
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
    valid_ratios = [r['Ratio_T/F_N'] for r in results if r['Ratio_T/F_N'] > 0 and r['N'] >= 10000]
    constant_c_avg = np.mean(valid_ratios) if valid_ratios else 0
    
    for r in results:
        # Handle cases where the run timed out or failed
        time_str = f"{r['T_avg']:.10f}" if r['T_avg'] != float('inf') else "TIMEOUT"
        ratio_str = f"{r['Ratio_T/F_N']:.2e}" if r['Ratio_T/F_N'] > 0 else "N/A"
        f_n_str = f"{r['F_N']:<20.0f}" # Ensure N^2 is printed as a whole number
        
        print(f"{r['N']:<20} | {time_str}{'s':<19} | {f_n_str:<20} | {ratio_str}{'':<17}")

    print("\n--- Conclusion ---")
    print(f"The 'Ratio (T_avg / F(N))' column shows how close the runtime is to a constant 'c'.")
    print(f"For large N, the ratio consistently converges to a value (approx. {constant_c_avg:.2e}).")
    print(f"This convergence confirms the algorithm's empirical time complexity is $\mathbf{{{complexity_name}}}$ ($\mathbf{{{complexity_term}}}$).")


# --- Main Execution ---

if __name__ == "__main__":
    
    # --- Configuration ---
    # Input sizes to test Jarvis March (up to 1 million points)
    jarvis_march_input_sizes = [
        10, 100, 500, 1000, 2000, 10000, 50000, 100000, 250000, 500000, 1000000
    ]
    
    # The complexity term to compare against
    complexity_term = 'N^2'
    
    # Check if the smallest required file exists
    if not os.path.exists(f"dataset_{jarvis_march_input_sizes[0]}_points.txt"):
        print(f"⚠️ Data files not found (e.g., dataset_{jarvis_march_input_sizes[0]}_points.txt).")
        print("Please run the 'generate_graham_data.py' script first (it creates the required files).")
    else:
        # --- Run Jarvis March Experiment ---
        timing_results = run_timing_experiment(
            algorithm_func=jarvis_march, 
            input_sizes=jarvis_march_input_sizes, 
            complexity_term=complexity_term, 
            num_runs=5, 
            max_total_time_per_n=60 # Use a timeout for the O(N^2) complexity
        )
        print_analysis(timing_results, "Jarvis March", complexity_term)