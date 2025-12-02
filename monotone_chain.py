import math

def dot_product(v1, v2, v3):

    return (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])

def area(convex_hull):
    sum1 = 0.0
    sum2 = 0.0
    for i in range(0,len(convex_hull)):
        sum1+=convex_hull[i][0]*convex_hull[(i+1)%len(convex_hull)][1]
    for i in range(0,len(convex_hull)):
        sum2+=convex_hull[i][1]*convex_hull[(i+1)%len(convex_hull)][0]
    return 0.5 * abs(sum1-sum2)

def perimeter(convex_hull):
    length = 0.0
    for i in range(0, len(convex_hull)):
        length += math.sqrt((convex_hull[i][0]-convex_hull[(i+1)%len(convex_hull)][0])**2+(convex_hull[i][1]-convex_hull[(i+1)%len(convex_hull)][1])**2)
    return length

def upper_hull_calc(vertex_list):
    upper_hull = []

    for i in range(0, len(vertex_list)):
        while len(upper_hull) >= 2 and dot_product(upper_hull[-2],upper_hull[-1],vertex_list[i]) <= 0:
            upper_hull.pop()
        upper_hull.append(vertex_list[i])

    return upper_hull


def lower_hull_calc(vertex_list):
    lower_hull = []

    for i in reversed(range(0, len(vertex_list))):
        while len(lower_hull)>=2 and dot_product(lower_hull[-2],lower_hull[-1], vertex_list[i]) <= 0:
            lower_hull.pop()
        lower_hull.append(vertex_list[i])

    return lower_hull


if __name__ == "__main__":
    # Your code
    vertex_list = [
        (1, 1),    # Inside
        (2, 5),    # Inside
        (3, 3),    # Inside
        (0, 0),    # Corner
        (5, 1),    # Inside
        (7, 2),    # Inside
        (10, 3),   # On hull
        (9, 6),    # Inside
        (6, 8),    # On hull
        (4, 7),    # Inside
        (1, 8),    # On hull
        (0, 5),    # Collinear on left edge
        (0, 3),    # Collinear on left edge
        (2, 2),    # Inside
        (4, 2),    # Inside
        (6, 3),    # Inside
        (8, 5),    # Inside
        (7, 7),    # Inside
        (5, 5),    # Inside
        (3, 6),    # Inside
        (10, 0),   # Collinear on bottom edge
        (8, 0),    # Collinear on bottom edge
        (3, 0),    # Collinear on bottom edge
        (1, 10),   # Corner (Top-left)
        (11, 6)   # Corner (Top-right)
    ]
    sorted_vertices = sorted(vertex_list, key=lambda item: item[0])
    upper_hull = upper_hull_calc(sorted_vertices)
    lower_hull = lower_hull_calc(sorted_vertices)
    convex_hull = lower_hull[:-1]+upper_hull[:-1]
    area_num = area(convex_hull)
    peri = perimeter(convex_hull)
    print(convex_hull)
    print("Area: ", area_num)
    print("Perimeter: ", peri)

