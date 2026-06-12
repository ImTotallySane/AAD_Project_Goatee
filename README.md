# AAD_Project_Goatee

This project implements various convex hull algorithms and a visualization tool for robot path planning using convex hull and Minkowski sum techniques.

# Installation

pip install -r requirements.txt

# Main Files
- app.py - Interactive GUI for robot path planning with convex hull and Minkowski sum visualization
- Code.ipynb - The main project, including the proofs, intuition, codes and visualization for these convex hull algorithms. Also included the heuristics and runtime analysis of these algorithms showing that they converge to the theoretical complexities.

# Scripts Directory
- generate_data.py - Generate random datasets for testing algorithms
- graham.py - Graham Scan algorithm with empirical complexity analysis
- jarvis.py - Jarvis March (Gift Wrapping) algorithm with complexity analysis
- monotone.py - Monotone Chain algorithm with complexity analysis
- quickhull.py - QuickHull algorithm with complexity analysis
- rand_quickhull.py - Randomized QuickHull with ray shooting technique
- inbuilt_convex.py - Scipy's built-in convex hull for comparison

## How to Run

First, generate random point datasets for algorithm testing:

->  python scripts/generate_data.py

This creates dataset files (e.g., `dataset_100_points.txt`, `dataset_1000_points.txt`, etc.)

Each algorithm script performs empirical complexity analysis:

# Graham Scan Algorithm (O(N log N))
python scripts/graham.py

# Jarvis March Algorithm (O(N²))
python scripts/jarvis.py

# Monotone Chain Algorithm (O(N log N))
python scripts/monotone.py

# QuickHull Algorithm (O(N log N) average case)
python scripts/quickhull.py

# Randomized QuickHull with Ray Shooting
python scripts/rand_quickhull.py

# Scipy Built-in Convex Hull
python scripts/inbuilt_convex.py

### 3. Run the Interactive Visualization Tool
Launch the GUI application for robot path planning:

python app.py


If `pyvisgraph` is not installed, pathfinding features will be disabled, but you can still draw obstacles and visualize convex hulls.

# App Usage Instructions:
- Obstacle Mode (Blue): Click to draw obstacle polygons, right-click to close the shape
- Robot Mode (Green): Draw your robot shape
- Start/End Points: Mark start (S) and end (E) points for pathfinding
- Calculate Path: Computes visibility graph and finds shortest path
- Spacebar: Save current configuration to `room_data.json`
- C key: Clear the entire canvas

## Algorithm Complexity Summary
- **Graham Scan**: O(N log N)
- **Monotone Chain**: O(N log N)
- **QuickHull**: O(N log N) average, O(N²) worst case
- **Jarvis March**: O(NH) where H is hull size, O(N²) worst case
- **Built-in (Qhull)**: O(N log N)