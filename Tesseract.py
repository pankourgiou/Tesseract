import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create vertices of a tesseract (4D hypercube)
def create_tesseract_vertices():
    vertices = []
    # All combinations of -1 and 1 in 4D
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                for w in [-1, 1]:
                    vertices.append([x, y, z, w])
    return np.array(vertices)

# Create edges: two vertices are connected if they differ in exactly one coordinate.
def create_tesseract_edges(vertices):
    edges = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            # If the sum of absolute differences equals 2, they differ in exactly one coordinate.
            if np.sum(np.abs(vertices[i] - vertices[j])) == 2:
                edges.append((i, j))
    return edges

# 4D rotation in the xw plane
def rotate_4d_xw(vertices, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c, 0, 0, -s],
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [s, 0, 0,  c]
    ])
    return vertices.dot(R.T)

# 4D rotation in the yw plane (for an additional folding effect)
def rotate_4d_yw(vertices, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [1,  0, 0,  0],
        [0,  c, 0, -s],
        [0,  0, 1,  0],
        [0,  s, 0,  c]
    ])
    return vertices.dot(R.T)

# Perspective projection from 4D to 3D
def project_4d_to_3d(vertices, distance=4):
    projected = []
    for v in vertices:
        # Using perspective: points with larger w appear "closer" or "farther"
        factor = distance / (distance - v[3])
        projected.append([v[0] * factor, v[1] * factor, v[2] * factor])
    return np.array(projected)

# Initialize vertices and edges
vertices = create_tesseract_vertices()
edges = create_tesseract_edges(vertices)

# Set up the Matplotlib 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])
ax.set_box_aspect([1, 1, 1])
ax.set_title("Folding and Unfolding Tesseract")

# Animation update function
def update(frame):
    ax.clear()
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Folding and Unfolding Tesseract")
    
    # Create oscillating angles to simulate folding/unfolding
    theta = 0.5 * np.sin(frame * 0.1)  # rotation in xw plane
    phi   = 0.5 * np.cos(frame * 0.1)  # rotation in yw plane

    # Apply rotations in 4D
    rotated = rotate_4d_xw(vertices, theta)
    rotated = rotate_4d_yw(rotated, phi)
    
    # Project the rotated 4D vertices to 3D
    projected = project_4d_to_3d(rotated, distance=4)
    
    # Draw each edge of the tesseract
    for edge in edges:
        start, end = edge
        xs = [projected[start, 0], projected[end, 0]]
        ys = [projected[start, 1], projected[end, 1]]
        zs = [projected[start, 2], projected[end, 2]]
        ax.plot(xs, ys, zs, color='blue', linewidth=1)
    
    # Optionally, draw the vertices as red dots
    ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], color='red')

# Create the animation: 200 frames with a 50ms interval between frames
ani = FuncAnimation(fig, update, frames=range(0, 200), interval=50)
plt.show()
