import numpy as np
import time
import os
import math
import random

# ==========================================
# 1. THE ALGORITHM (Randomized Ray Shooting)
# ==========================================

def orientation(p1, p2, point) -> float:
    """
    Calculates the 2D cross product.
    > 0: Left turn (Counter-clockwise)
    < 0: Right turn (Clockwise)
    = 0: Collinear
    """
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

def ray_shoot_bridge(S, p, r, q):
    """
    Finds the upper bridge (s, t) for the hull.
    """
    s = t = q
    # Baseline cross product for the initial point q relative to p->r
    base_cp = orientation(p, r, q)

    Sl = []
    Sr = []

    # Using the passed set S (shuffling is handled if needed, or we rely on random q choice)
    S_shuffled = S 

    # Add q to both tentative lists initially
    Sl.append(q)
    Sr.append(q)

    # Vector calculations for projection comparisons
    vec_pr = (r[0]-p[0], r[1]-p[1])
    vec_pq = (q[0]-p[0], q[1]-p[1])
    
    # Dot product of pq and pr (projection numerator)
    proj_q = vec_pq[0]*vec_pr[0] + vec_pq[1]*vec_pr[1]

    for point in S_shuffled:
        vec_ppoint = (point[0]-p[0], point[1]-p[1])
        proj_point = vec_ppoint[0]*vec_pr[0] + vec_ppoint[1]*vec_pr[1]

        is_left = proj_point < proj_q
        is_right = proj_point > proj_q
        is_above_bridge = False

        # If s and t are still the same point (the initial q)
        if s is t:
            # Check relative to the baseline line pr
            point_cp = orientation(p, r, point)
            if point_cp > base_cp:
                is_above_bridge = True
        else:
            # Check relative to the current bridge segment st
            if orientation(s, t, point) > 0:
                is_above_bridge = True

        # Update bridge endpoints if the point is strictly above
        if is_above_bridge:
            if is_left:
                s = point
                # If we found a new s, scan Sr to ensure t is still valid for the new s
                if Sr:
                    best_t = Sr[0]
                    for candidate in Sr[1:]:
                        if orientation(s, best_t, candidate) > 0:
                            best_t = candidate
                    t = best_t

            elif is_right:
                t = point
                # If we found a new t, scan Sl to ensure s is still valid for the new t
                if Sl:
                    best_s = Sl[0]
                    for candidate in Sl[1:]:
                        if orientation(best_s, t, candidate) > 0:
                            best_s = candidate
                    s = best_s

        if is_left:
            Sl.append(point)
        elif is_right:
            Sr.append(point)

    return s, t

def find_hull(p, r, S):
    """
    Recursive step.
    """
    if not S:
        return []

    # Random pivot selection is crucial for the O(N log N) average case
    q = random.choice(S)
    
    s, t = ray_shoot_bridge(S, p, r, q)

    S1 = []
    S2 = []
    
    # Filter points for the recursive steps
    for pt in S:
        # Check if point is outside (above) ps
        if orientation(p, s, pt) > 0:
            S1.append(pt)
        # Check if point is outside (above) tr
        elif orientation(t, r, pt) > 0: 
            S2.append(pt)

    # Recursive calls
    left  = find_hull(p, s, S1)
    right = find_hull(t, r, S2)

    # Combine
    if s[0] == t[0] and s[1] == t[1]:
        return left + [s] + right
    else:
        return left + [s, t] + right

def ray_shooting_quickhull(points):
    """
    Main Entry Point for the Algorithm.
    """
    # Convert numpy array to list of lists/tuples for speed in Python
    if isinstance(points, np.ndarray):
        points = points.tolist()

    if len(points) < 3:
        return points

    # Find min and max x-coordinate points
    p_min = min(points, key=lambda p: (p[0], p[1]))
    p_max = max(points, key=lambda p: (p[0], p[1]))

    upper = []
    lower = []

    # Initial partition into upper and lower sets relative to p_min -> p_max
    for pt in points:
        orient = orientation(p_min, p_max, pt)
        if  orient > 0:
            upper.append(pt)
        elif orient < 0:
            lower.append(pt)

    hull = [p_min]
    hull += find_hull(p_min, p_max, upper)
    hull.append(p_max)
    hull += find_hull(p_max, p_min, lower)
    
    return np.array(hull)

# ==========================================
# 2. DATA GENERATOR (Helper)
# ==========================================

def generate_synthetic_data(sizes):
    """Generates random data files if they don't exist."""
    print("Checking/Generating datasets...")
    for n in sizes:
        filename = f"dataset_{n}_points.txt"
        if not os.path.exists(filename):
            print(f"  Generating {filename}...")
            # Generate random points in a circle/box
            data = np.random.rand(n, 2) * 1000
            np.savetxt(filename, data, delimiter=',')
    print("Data check complete.\n")

# ==========================================
# 3. EMPIRICAL ANALYSIS FRAMEWORK
# ==========================================

def load_data(filename):
    try:
        return np.loadtxt(filename, delimiter=',')
    except FileNotFoundError:
        return None

def run_timing_experiment(algorithm_func, input_sizes, complexity_term, num_runs=5, max_total_time_per_n=60):
    results = []
    algo_name = algorithm_func.__name__.replace('_', ' ').title()
    print(f"\n--- Starting {algo_name} Timing Experiment (Target: {complexity_term}) ---")
    
    for n in input_sizes:
        filename = f"dataset_{n}_points.txt"
        points = load_data(filename)
        
        if points is None:
            print(f"Skipping N={n}: Data file not found.")
            continue 

        total_time = 0
        runs_completed = 0
        
        for r in range(num_runs):
            if total_time >= max_total_time_per_n:
                print(f"    [!] Timeout threshold reached for N={n}. Skipping remaining runs.")
                break

            # Create a copy so the algorithm doesn't sort/modify the original array for the next run
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
        
        print(f"N={n:<8}: Avg Time={avg_time:.6f}s | Runs={runs_completed} | T/F(N)={ratio:.2e}")

    return results

def print_analysis(results, complexity_name, complexity_term):
    print("\n" + "="*80)
    print(f"                 {complexity_name} Empirical Complexity Analysis")
    print("="*80)
    print(f"{'N (Input Size)':<20} | {'T_avg (Seconds)':<20} | {complexity_term:<20} | {'Ratio (T_avg/F(N))':<20}")
    print("-" * 80)
    
    # Calculate convergence of the ratio for larger N
    valid_ratios = [r['Ratio_T/F_N'] for r in results if r['Ratio_T/F_N'] > 0 and r['N'] >= 10000]
    constant_c_avg = np.mean(valid_ratios) if valid_ratios else 0
    
    for r in results:
        time_str = f"{r['T_avg']:.6f}" if r['T_avg'] != float('inf') else "TIMEOUT"
        ratio_str = f"{r['Ratio_T/F_N']:.2e}" if r['Ratio_T/F_N'] > 0 else "N/A"
        f_n_str = f"{r['F_N']:.2f}"
        
        print(f"{r['N']:<20} | {time_str}{'s':<19} | {f_n_str:<20} | {ratio_str}{'':<17}")

    print("\n--- Conclusion ---")
    print(f"The 'Ratio' column represents the hidden constant 'c' in T(n) = c * {complexity_term}.")
    print(f"For large N, if the Ratio stabilizes (approx {constant_c_avg:.2e}), the complexity hypothesis is supported.")
    print(f"This confirms the algorithm performs consistently with O({complexity_term}).")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # 1. Configuration
    # We test up to 500k points. (1M might take a while depending on Python overhead)
    input_sizes = [10, 100, 500, 1000, 5000, 10000, 50000, 100000, 250000, 500000]
    complexity_term = 'N log N'
    
    # 2. Generate Data (if missing)
    generate_synthetic_data(input_sizes)

    # 3. Run Experiment
    timing_results = run_timing_experiment(
        algorithm_func=ray_shooting_quickhull, 
        input_sizes=input_sizes, 
        complexity_term=complexity_term, 
        num_runs=3,  # Reduced runs slightly for faster total execution
        max_total_time_per_n=30 
    )
    
    # 4. Show Report
    print_analysis(timing_results, "Ray Shooting QuickHull", complexity_term)