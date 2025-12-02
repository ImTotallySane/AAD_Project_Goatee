import numpy as np
import os

def generate_convex_hull_dataset(n_points, filename=None):

    if n_points <= 0:
        print("Error: Number of points must be greater than 0.")
        return

    X = np.random.uniform(low=-1000.0, high=1000.0, size=n_points)
    
    Y = np.random.uniform(low=-1000.0, high=1000.0, size=n_points)

    data = np.stack((X, Y), axis=1)

    if filename is None:
        filename = f"dataset_{n_points}_points.txt"
    try:
        np.savetxt(filename, data, fmt='%.4f', delimiter=',')
        print(f"✅ Generated {n_points} points in range [-1000, 1000]. File: {filename}")
    except Exception as e:
        print(f"An error occurred while saving the file (possibly memory-related for large N): {e}")

input_sizes = [
    10,
    100,
    500,
    1000,
    2000,
    10000,
    50000,
    100000,
    250000,
    500000,
    1000000,
]

print("Starting dataset generation (X, Y only, range [-1000, 1000], 4 decimal places)...")
for n in input_sizes:
    generate_convex_hull_dataset(n)

print("\nAll required datasets have been successfully generated.")