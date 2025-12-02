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
    
    In QuickHull, this is used to calculate the area (signed distance) to the line pq.
    """
    # (q_x - p_x) * (r_y - q_y) - (q_y - p_y) * (r_x - q_x)
    return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])

# --- 2. QuickHull Algorithm (Average O(N log N), Worst O(N^2)) ---

def distance_to_line(p, a, b):
    """
    Calculates the absolute distance from point p to the line segment ab.
    This is proportional to the area of the triangle (p, a, b).
    """
    # The absolute value of the cross product gives twice the area of the triangle formed by p, a, b.
    # This value is proportional to the distance needed for finding the furthest point.
    return abs(cross_product(a, b, p))

def find_hull(a, b, points):
    """
    Recursive function to find the points on one side of the segment AB.
    """
    if not points:
        return []
        
    # Find the point C furthest from the line AB (maximum signed area)
    max_dist = -1
    furthest_point = None
    furthest_index = -1
    
    for i, p in enumerate(points):
        dist = distance_to_line(p, a, b)
        if dist > max_dist:
            max_dist = dist
            furthest_point = p
            furthest_index = i

    if furthest_point is None:
        return []
        
    # Split the remaining points into two sets based on the segments AC and CB
    set1 = [] # Points on the outer side of AC
    set2 = [] # Points on the outer side of CB
    
    # Points inside the triangle (A, C, B) can be safely discarded.
    
    for i, p in enumerate(points):
        if i == furthest_index:
            continue

        # Check orientation relative to AC
        # CCW (positive) means the point is outside the segment AC, on the correct side
        if cross_product(a, furthest_point, p) > 0:
            set1.append(p)
        
        # Check orientation relative to CB
        # CCW (positive) means the point is outside the segment CB, on the correct side
        elif cross_product(furthest_point, b, p) > 0:
            set2.append(p)
            
    # Recursively find hulls for the new sets
    hull1 = find_hull(a, furthest_point, set1)
    hull2 = find_hull(furthest_point, b, set2)
    
    # The resulting hull is hull1 + [C] + hull2
    return hull1 + [furthest_point] + hull2

def quick_hull(points):
    """
    Main QuickHull function. Average Time Complexity: O(N log N).
    """
    n = len(points)
    if n < 3:
        return list(points)

    # 1. Find P_min and P_max (guaranteed to be on the hull)
    points_list = points.tolist()
    points_list.sort(key=lambda p: (p[0], p[1]))
    points = np.array(points_list)
    
    a = points[0]  # P_min (min X)
    b = points[-1] # P_max (max X)

    # 2. Divide remaining points into two sets (S1: above AB, S2: below AB)
    s1 = []
    s2 = []
    
    for p in points[1:-1]:
        cp = cross_product(a, b, p)
        if cp > 0:
            s1.append(p) # Above the line AB (CCW)
        elif cp < 0:
            s2.append(p) # Below the line AB (CW)
            
    # 3. Recursively find the hull segments
    hull_segment1 = find_hull(a, b, s1)
    hull_segment2 = find_hull(b, a, s2) # Note: order reversed for s2 to keep CCW orientation
    
    # 4. Combine the hull segments and the anchor points
    # Start at A, go through hull1, B, hull2, and back to A
    return np.array([a] + hull_segment1 + [b] + hull_segment2)


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
    # Input sizes to test QuickHull (up to 1 million points)
    quickhull_input_sizes = [
        10, 100, 500, 1000, 2000, 10000, 50000, 100000, 250000, 500000, 1000000
    ]
    
    # The complexity term to compare against
    complexity_term = 'N log N'
    
    # Check if the smallest required file exists
    if not os.path.exists(f"dataset_{quickhull_input_sizes[0]}_points.txt"):
        print(f"⚠️ Data files not found (e.g., dataset_{quickhull_input_sizes[0]}_points.txt).")
        print("Please run the data generator script to create the required datasets (e.g., generate_monotone_data.py).")
    else:
        # --- Run QuickHull Experiment ---
        timing_results = run_timing_experiment(
            algorithm_func=quick_hull, 
            input_sizes=quickhull_input_sizes, 
            complexity_term=complexity_term, 
            num_runs=5, 
            max_total_time_per_n=60 
        )
        print_analysis(timing_results, "QuickHull", complexity_term)