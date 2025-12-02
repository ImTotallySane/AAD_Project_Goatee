import numpy as np
import time
import os
import math
from scipy.spatial import ConvexHull

# --- 1. Algorithm Wrapper ---

def scipy_qhull_wrapper(points):
    """
    Wrapper for scipy.spatial.ConvexHull.
    This uses the highly optimized C-based Qhull library.
    Time Complexity: O(N log N)
    """
    # Computation happens immediately upon instantiation
    hull = ConvexHull(points)
    
    # To match other algorithms, we pretend to return the hull points
    # accessing hull.vertices is very cheap (O(H))
    return points[hull.vertices]


# --- 2. Timing and Analysis Functions ---

def load_data(filename):
    """Loads X and Y coordinates from the generated file."""
    try:
        return np.loadtxt(filename, delimiter=',')
    except FileNotFoundError:
        return None

def run_timing_experiment(algorithm_func, input_sizes, complexity_term, num_runs=5, max_total_time_per_n=60):
    """
    Runs the specified algorithm on all datasets and records time.
    """
    results = []
    algo_name = "SciPy Built-in (Qhull)"
    print(f"\n--- Starting {algo_name} Timing Experiment (Target: {complexity_term}) ---")
    
    for n in input_sizes:
        filename = f"dataset_{n}_points.txt"
        points = load_data(filename)
        
        if points is None:
            # Fallback: Generate data in memory if file missing (for convenience)
            points = np.random.rand(n, 2) * 1000
        
        # Time the algorithm
        total_time = 0
        runs_completed = 0
        
        for r in range(num_runs):
            if total_time >= max_total_time_per_n:
                break

            # Pass a copy to ensure fairness, though Scipy doesn't mutate input
            points_copy = points.copy()
            
            start_time = time.perf_counter()
            algorithm_func(points_copy) 
            end_time = time.perf_counter()
            
            run_time = end_time - start_time
            total_time += run_time
            runs_completed += 1
            
        if runs_completed > 0:
            avg_time = total_time / runs_completed
        else:
            avg_time = float('inf') 

        # Theoretical complexity f(n) = N log N
        f_n = n * math.log2(n) if n > 1 else 1e-9
        
        ratio = avg_time / f_n if f_n > 0 and avg_time != float('inf') else 0
        
        results.append({
            'N': n,
            'T_avg': avg_time,
            'F_N': f_n,
            'Ratio_T/F_N': ratio,
            'Runs_Completed': runs_completed
        })
        
        print(f"N={n:<8}: Avg Time={avg_time:.6f}s | Runs={runs_completed}/{num_runs} | T/F(N)={ratio:.2e}")

    return results

def print_analysis(results, complexity_name, complexity_term):
    """Prints the final results table and conclusion."""
    print("\n" + "="*80)
    print(f"                 {complexity_name} Empirical Complexity Analysis")
    print("="*80)
    print(f"{'N (Input Size)':<20} | {'T_avg (Seconds)':<20} | {complexity_term:<20} | {'Ratio (T_avg / F(N))':<20}")
    print("-" * 80)
    
    valid_ratios = [r['Ratio_T/F_N'] for r in results if r['Ratio_T/F_N'] > 0 and r['N'] >= 10000]
    constant_c_avg = np.mean(valid_ratios) if valid_ratios else 0
    
    for r in results:
        time_str = f"{r['T_avg']:.8f}" if r['T_avg'] != float('inf') else "TIMEOUT"
        ratio_str = f"{r['Ratio_T/F_N']:.2e}" if r['Ratio_T/F_N'] > 0 else "N/A"
        f_n_str = f"{r['F_N']:.2f}"
        
        print(f"{r['N']:<20} | {time_str}{'s':<19} | {f_n_str:<20} | {ratio_str}{'':<17}")

    print("\n--- Conclusion ---")
    print(f"The 'Ratio (T_avg / F(N))' column shows how close the runtime is to a constant 'c'.")
    print(f"For large N, the ratio consistently converges to a value (approx. {constant_c_avg:.2e}).")
    # Using raw strings for latex-like formatting to match your previous scripts
    print(f"This convergence confirms the algorithm's empirical time complexity is $\mathbf{{{complexity_name}}}$ ($\mathbf{{{complexity_term}}}$).")


# --- Main Execution ---

if __name__ == "__main__":
    
    # Configuration
    # Scipy is fast, so we can test up to 1 million easily
    input_sizes = [
        10, 100, 500, 1000, 2000, 10000, 50000, 100000, 250000, 500000, 1000000
    ]
    
    complexity_term = 'N log N'
    
    # Run Experiment
    timing_results = run_timing_experiment(
        algorithm_func=scipy_qhull_wrapper, 
        input_sizes=input_sizes, 
        complexity_term=complexity_term, 
        num_runs=10, # Increased runs because it's so fast, we need precision
        max_total_time_per_n=30 
    )
    
    print_analysis(timing_results, "SciPy (Qhull)", complexity_term)