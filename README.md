# AAD_Project_Goatee

## Description
This project implements various convex hull algorithms and a visualization tool for robot path planning using convex hull and Minkowski sum techniques.

## Prerequisites
- Python 3.7+
- pip (Python package installer)

## Installation

### Install Required Dependencies
```bash
pip install numpy scipy pyvisgraph
```

Or install all dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

### Core Dependencies
- **numpy** - For numerical operations and array handling
- **scipy** - For built-in convex hull algorithm (ConvexHull)
- **pyvisgraph** - For visibility graph pathfinding (optional, required for `app.py`)
- **tkinter** - For GUI (usually comes pre-installed with Python)

## Project Structure

### Main Files
- **app.py** - Interactive GUI for robot path planning with convex hull and Minkowski sum visualization
- **monotone_chain.py** - Standalone monotone chain convex hull implementation

### Scripts Directory
- **generate_data.py** - Generate random datasets for testing algorithms
- **graham.py** - Graham Scan algorithm with empirical complexity analysis
- **jarvis.py** - Jarvis March (Gift Wrapping) algorithm with complexity analysis
- **monotone.py** - Monotone Chain algorithm with complexity analysis
- **quickhull.py** - QuickHull algorithm with complexity analysis
- **rand_quickhull.py** - Randomized QuickHull with ray shooting technique
- **inbuilt_convex.py** - Scipy's built-in convex hull for comparison

## How to Run

### 1. Generate Test Datasets
First, generate random point datasets for algorithm testing:
```bash
python scripts/generate_data.py
```
This creates dataset files (e.g., `dataset_100_points.txt`, `dataset_1000_points.txt`, etc.)

### 2. Run Individual Convex Hull Algorithms
Each algorithm script performs empirical complexity analysis:

```bash
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
```

### 3. Run the Interactive Visualization Tool
Launch the GUI application for robot path planning:
```bash
python app.py
```

**Note**: If `pyvisgraph` is not installed, pathfinding features will be disabled, but you can still draw obstacles and visualize convex hulls.

#### App Usage Instructions:
- **Obstacle Mode (Blue)**: Click to draw obstacle polygons, right-click to close the shape
- **Robot Mode (Green)**: Draw your robot shape
- **Start/End Points**: Mark start (S) and end (E) points for pathfinding
- **Calculate Path**: Computes visibility graph and finds shortest path
- **Spacebar**: Save current configuration to `room_data.json`
- **C key**: Clear the entire canvas

### 4. Run Standalone Monotone Chain
```bash
python monotone_chain.py
```
This runs a demo with hardcoded test points and calculates the convex hull area and perimeter.

## Algorithm Complexity Summary
- **Graham Scan**: O(N log N)
- **Monotone Chain**: O(N log N)
- **QuickHull**: O(N log N) average, O(N²) worst case
- **Jarvis March**: O(NH) where H is hull size, O(N²) worst case
- **Built-in (Qhull)**: O(N log N)

## Output Files
- `room_data.json` - Saved configuration from the GUI app (obstacles, hulls, robot shape, path)
- `dataset_*_points.txt` - Generated test datasets
