import tkinter as tk
import json
import monotone_chain

try:
    from pyvisgraph import VisGraph, Point
    PYVISGRAPH_AVAILABLE = True
except ImportError:
    PYVISGRAPH_AVAILABLE = False
    print("Warning: 'pyvisgraph' not found. Pathfinding features will be disabled.")
    print("Install it using: pip install pyvisgraph")

class ShapeDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Convex Hull Robot Setup (Minkowski Sum + VisGraph)")
        
        # Data storage
        self.obstacles = []     # List of obstacle polygons (raw)
        self.hulls = []         # List of obstacle hulls (Red lines)
        self.cspace_hulls = []  # List of Minkowski Sum hulls (Purple lines)
        
        self.robot_shape = []   # The robot polygon
        self.start_point = None # (x, y)
        self.end_point = None   # (x, y)
        
        # Graph Data
        self.vis_graph_edges = [] # List of ((x1, y1), (x2, y2))
        self.shortest_path = []   # List of (x, y) points
        
        self.current_shape = [] # Points of the shape currently being drawn
        self.mode = tk.StringVar(value="obstacle") # Modes: obstacle, robot, start, end

        # --- UI LAYOUT ---
        
        # 1. Top Control Panel
        control_frame = tk.Frame(root, bg="#e0e0e0", pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(control_frame, text="Draw Mode:", bg="#e0e0e0", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        
        modes = [
            ("Obstacle (Blue)", "obstacle"),
            ("Robot Shape (Green)", "robot"),
            ("Start Point (S)", "start"),
            ("End Point (E)", "end")
        ]
        
        for text, mode_val in modes:
            rb = tk.Radiobutton(
                control_frame, text=text, variable=self.mode, value=mode_val,
                bg="#e0e0e0", fg="black", selectcolor="#ffffff",
                command=self.reset_current_shape
            )
            rb.pack(side=tk.LEFT, padx=5)

        # 2. Pathfinding Button
        self.btn_calc = tk.Button(
            control_frame, text="Calculate Path", 
            bg="#4CAF50", fg="black", font=("Arial", 9, "bold"),
            command=self.calculate_visibility_path
        )
        self.btn_calc.pack(side=tk.LEFT, padx=20)
        
        if not PYVISGRAPH_AVAILABLE:
            self.btn_calc.config(state="disabled", text="pyvisgraph missing")

        # 3. Canvas
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 4. Instructions
        instruction_text = (
            "INSTRUCTIONS:\n"
            "• Obstacle (Blue): Draw obstacles. Red=Hull, Purple=C-Space.\n"
            "• Robot (Green): Draw robot to update C-Space. Start(S)/End(E) for path.\n"
            "• 'Calculate Path': Merges overlaps -> Builds Graph -> Finds Path.\n"
            "• Spacebar: Save JSON | C: Clear Board"
        )
        self.label = tk.Label(
            root, text=instruction_text, justify=tk.LEFT, 
            font=("Consolas", 10), bg="#333333", fg="white",
            relief="solid", borderwidth=1
        )
        self.label.pack(fill=tk.X, ipady=5)

        # Event Bindings
        self.canvas.bind("<Button-1>", self.handle_click)    # Left Click
        self.canvas.bind("<Button-3>", self.close_polygon)   # Right Click (Windows/Linux)
        self.canvas.bind("<Button-2>", self.close_polygon)   # Right Click (MacOS)
        self.root.bind("<space>", self.export_data)          # Spacebar
        self.root.bind("c", self.clear_board)                # 'c' key

    def reset_current_shape(self):
        self.current_shape = []
        self.redraw_all()

    def handle_click(self, event):
        x, y = event.x, event.y
        mode = self.mode.get()

        if mode == "obstacle" or mode == "robot":
            self.add_poly_point(x, y, mode)
        elif mode == "start":
            self.start_point = (x, y)
            # Invalidate old path when points change
            self.vis_graph_edges = []
            self.shortest_path = []
            self.redraw_all()
        elif mode == "end":
            self.end_point = (x, y)
            # Invalidate old path when points change
            self.vis_graph_edges = []
            self.shortest_path = []
            self.redraw_all()

    def add_poly_point(self, x, y, mode):
        self.current_shape.append((x, y))
        color = "blue" if mode == "obstacle" else "green"
        
        r = 3
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=color)
        
        if len(self.current_shape) > 1:
            last_x, last_y = self.current_shape[-2]
            self.canvas.create_line(last_x, last_y, x, y, fill="black", width=2)

    def close_polygon(self, event):
        mode = self.mode.get()
        if mode not in ["obstacle", "robot"]:
            return

        if len(self.current_shape) < 3:
            print("Need at least 3 points to make a shape!")
            return
            
        if mode == "obstacle":
            self.obstacles.append(self.current_shape)
            hull = self.calculate_hull(self.current_shape)
            self.hulls.append(hull)
            
            if self.robot_shape:
                ms_hull = self.calculate_minkowski_hull(hull, self.robot_shape)
                self.cspace_hulls.append(ms_hull)
            else:
                self.cspace_hulls.append([])
            
        elif mode == "robot":
            self.robot_shape = self.current_shape
            self.cspace_hulls = []
            for obs_hull in self.hulls:
                ms_hull = self.calculate_minkowski_hull(obs_hull, self.robot_shape)
                self.cspace_hulls.append(ms_hull)
        
        # Invalidate old path when geometry changes
        self.vis_graph_edges = []
        self.shortest_path = []
            
        self.current_shape = []
        self.redraw_all()

    def calculate_hull(self, points):
        sorted_vertices = sorted(points, key=lambda item: item[0])
        upper = monotone_chain.upper_hull_calc(sorted_vertices)
        lower = monotone_chain.lower_hull_calc(sorted_vertices)
        return lower[:-1] + upper[:-1]

    def calculate_minkowski_hull(self, obstacle_pts, robot_pts):
        if not robot_pts or not obstacle_pts:
            return []

        rx_sum = sum(p[0] for p in robot_pts)
        ry_sum = sum(p[1] for p in robot_pts)
        rc_x = rx_sum / len(robot_pts)
        rc_y = ry_sum / len(robot_pts)

        cloud = []
        for o_pt in obstacle_pts:
            for r_pt in robot_pts:
                r_vec_x = r_pt[0] - rc_x
                r_vec_y = r_pt[1] - rc_y
                new_x = o_pt[0] - r_vec_x
                new_y = o_pt[1] - r_vec_y
                cloud.append((new_x, new_y))

        return self.calculate_hull(cloud)

    # --- GEOMETRY HELPER FUNCTIONS FOR INTERSECTION ---
    def on_segment(self, p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0
        return 1 if val > 0 else 2

    def do_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4: return True
        if o1 == 0 and self.on_segment(p1, p2, q1): return True
        if o2 == 0 and self.on_segment(p1, q2, q1): return True
        if o3 == 0 and self.on_segment(p2, p1, q2): return True
        if o4 == 0 and self.on_segment(p2, q1, q2): return True
        return False

    def is_point_in_poly(self, point, poly):
        # Ray casting algorithm
        x, y = point
        inside = False
        n = len(poly)
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def polygons_intersect(self, poly1, poly2):
        # 1. Check if any edges intersect
        n1 = len(poly1)
        n2 = len(poly2)
        for i in range(n1):
            for j in range(n2):
                if self.do_intersect(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                    return True
        
        # 2. Check if poly1 is inside poly2 (check first point)
        if n1 > 0 and self.is_point_in_poly(poly1[0], poly2):
            return True
            
        # 3. Check if poly2 is inside poly1 (check first point)
        if n2 > 0 and self.is_point_in_poly(poly2[0], poly1):
            return True
            
        return False

    def merge_overlapping_hulls(self, hulls_list):
        # Iteratively merge intersecting hulls until no intersections remain
        # This keeps the result as a list of Convex Hulls
        current_hulls = list(hulls_list)
        
        while True:
            merged = False
            new_list = []
            skip_indices = set()
            
            for i in range(len(current_hulls)):
                if i in skip_indices: continue
                
                poly_a = current_hulls[i]
                
                # Check A against all remaining polys for overlap
                merged_with_poly = None
                
                for j in range(i + 1, len(current_hulls)):
                    if j in skip_indices: continue
                    poly_b = current_hulls[j]
                    
                    if self.polygons_intersect(poly_a, poly_b):
                        # Merge A and B -> New Hull
                        combined_cloud = poly_a + poly_b
                        merged_with_poly = self.calculate_hull(combined_cloud)
                        skip_indices.add(j)
                        merged = True
                        break # Break inner loop to process the merged result in next big iteration or logic
                
                if merged_with_poly:
                    new_list.append(merged_with_poly)
                else:
                    new_list.append(poly_a)
            
            current_hulls = new_list
            if not merged:
                break
                
        return current_hulls

    def calculate_visibility_path(self):
        if not PYVISGRAPH_AVAILABLE:
            print("Error: pyvisgraph not installed.")
            return

        if not self.start_point or not self.end_point:
            print("Set Start and End points first!")
            return

        print("Processing Hulls...")
        
        # 1. Merge overlapping Minkowski Hulls
        # Use a copy of the list so we don't destroy the original disjoint visualizations if not desired
        # But for the graph, we need the merged version.
        merged_cspace_hulls = self.merge_overlapping_hulls(self.cspace_hulls)
        print(f"Original Hulls: {len(self.cspace_hulls)} -> Merged Hulls: {len(merged_cspace_hulls)}")

        # 2. Convert to PyVisGraph Polygons
        graph_polys = []
        for hull in merged_cspace_hulls:
            if len(hull) < 3: continue
            poly_points = [Point(x, y) for x, y in hull]
            graph_polys.append(poly_points)

        # 3. Build Graph
        print("Building Visibility Graph...")
        graph = VisGraph()
        graph.build(graph_polys)

        # 4. Extract Edges for Visualization
        self.vis_graph_edges = []
        drawn_edges = set()

        for point, edges in graph.visgraph.graph.items():
            for edge in edges:
                p1, p2 = edge.p1, edge.p2
                edge_id = tuple(sorted(((p1.x, p1.y), (p2.x, p2.y))))
                if edge_id not in drawn_edges:
                    self.vis_graph_edges.append(((p1.x, p1.y), (p2.x, p2.y)))
                    drawn_edges.add(edge_id)

        # 5. Find Shortest Path
        s = Point(self.start_point[0], self.start_point[1])
        e = Point(self.end_point[0], self.end_point[1])
        
        try:
            shortest = graph.shortest_path(s, e)
            self.shortest_path = [(p.x, p.y) for p in shortest]
            print(f"Path Found: {len(self.shortest_path)} points.")
        except Exception as err:
            print(f"Pathfinding Error: {err}")
            self.shortest_path = []

        # We also want to visualize the MERGED hulls that the graph used, 
        # but maybe just keep the original purple ones for the "Obstacle View" 
        # and show the path over them. 
        # If you want to SEE the merged result, uncomment the next line:
        # self.cspace_hulls = merged_cspace_hulls 
        
        self.redraw_all()

    def redraw_all(self):
        self.canvas.delete("all")
        
        # 1. Draw Visibility Graph (Light Black/Gray Lines)
        for p1, p2 in self.vis_graph_edges:
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#cccccc", width=1)

        # 2. Draw Obstacles & Hulls
        for i, shape in enumerate(self.obstacles):
            self.canvas.create_polygon(shape, fill="light blue", outline="blue", stipple="gray50")
            
            if i < len(self.hulls):
                hull = self.hulls[i]
                if len(hull) > 0:
                    for j in range(len(hull)):
                        p1 = hull[j]
                        p2 = hull[(j + 1) % len(hull)]
                        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="red", width=2)

            if i < len(self.cspace_hulls):
                ms_hull = self.cspace_hulls[i]
                if len(ms_hull) > 0:
                    for j in range(len(ms_hull)):
                        p1 = ms_hull[j]
                        p2 = ms_hull[(j + 1) % len(ms_hull)]
                        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="purple", width=3, dash=(4, 4))

        # 3. Draw Robot
        if self.robot_shape:
            self.canvas.create_polygon(self.robot_shape, fill="light green", outline="green")
            cx = sum(p[0] for p in self.robot_shape) / len(self.robot_shape)
            cy = sum(p[1] for p in self.robot_shape) / len(self.robot_shape)
            self.canvas.create_text(cx, cy, text="ROBOT", font=("Arial", 8, "bold"), fill="dark green")

        # 4. Draw Shortest Path (Thick Green Line)
        if len(self.shortest_path) > 1:
            for i in range(len(self.shortest_path) - 1):
                p1 = self.shortest_path[i]
                p2 = self.shortest_path[i+1]
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#00ff00", width=4, capstyle=tk.ROUND)

        # 5. Draw Start/End
        if self.start_point:
            x, y = self.start_point
            r = 6
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="green", outline="black")
            self.canvas.create_text(x, y, text="S", fill="white", font=("Arial", 8, "bold"))

        if self.end_point:
            x, y = self.end_point
            r = 6
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="black")
            self.canvas.create_text(x, y, text="E", fill="white", font=("Arial", 8, "bold"))

        # 6. Active Drawing Lines
        if self.current_shape:
            color = "blue" if self.mode.get() == "obstacle" else "green"
            for i, (x, y) in enumerate(self.current_shape):
                r = 3
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=color)
                if i > 0:
                    lx, ly = self.current_shape[i-1]
                    self.canvas.create_line(lx, ly, x, y, fill="black")

    def export_data(self, event):
        data = {
            "obstacles": self.obstacles,
            "convex_hulls": self.hulls,
            "cspace_hulls": self.cspace_hulls,
            "robot": self.robot_shape,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "shortest_path": self.shortest_path
        }
        
        with open("room_data.json", "w") as f:
            json.dump(data, f, indent=4)
        
        print("Successfully saved to 'room_data.json'")
        self.label.config(text="Saved to room_data.json!", bg="light green", fg="black")
        self.root.after(2000, lambda: self.label.config(text="INSTRUCTIONS: See top panel for modes. Space to Save.", bg="#333333", fg="white"))

    def clear_board(self, event):
        self.obstacles = []
        self.hulls = []
        self.cspace_hulls = []
        self.robot_shape = []
        self.start_point = None
        self.end_point = None
        self.vis_graph_edges = []
        self.shortest_path = []
        self.current_shape = []
        self.redraw_all()
        print("Canvas Cleared")

if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeDrawer(root)
    root.mainloop()
