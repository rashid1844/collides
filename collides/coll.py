import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt


@njit
def _rotation_matrix(theta):
    """2d Rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

@njit
def _corner_points(center, theta, L, W):
    """finds four corners of vehicle, based on center, theta, L, W"""
    L_vec = np.array([L / 2 * np.cos(theta), L / 2 * np.sin(theta)])
    W_vec = np.array([-W / 2 * np.sin(theta), W / 2 * np.cos(theta)])
    corners = [center + L_vec + W_vec, center + L_vec - W_vec, center - L_vec - W_vec, center - L_vec + W_vec]
    return corners

@njit
def _corner_collision(ref_center, ref_theta, ref_L, ref_W, corners):
    """Checks if corners are within vehicle ref"""
    R = _rotation_matrix(-ref_theta)  # -theta: to set theta to zero
    corners = R @ (corners[0] - ref_center), R @ (corners[1] - ref_center), R @ (corners[2] - ref_center), R @ (corners[3] - ref_center)
    for c in corners:
        if -ref_L / 2 <= c[0] <= ref_L / 2 and -ref_W / 2 <= c[1] <= ref_W / 2:
            return True
    return False

@njit
def _collision(center1, theta1, L1, W1, center2, theta2, L2, W2):
    """collision of two sampling points with dimentions"""
    dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    if dist > np.sqrt((L1 / 2) ** 2 + (W1 / 2) ** 2) + np.sqrt((L2 / 2) ** 2 + (W2 / 2) ** 2):
        return False
    if dist <= W1 / 2 + W2 / 2:
        return True
    Acorners = _corner_points(center1, theta1, L1, W1)
    Bcorners = _corner_points(center2, theta2, L2, W2)
    collision_bool = _corner_collision(center1, theta1, L1, W1, Bcorners) or _corner_collision(center2, theta2, L2, W2, Acorners)
    return collision_bool

def collide(x1, y1, x2, y2, L1, W1, theta1, L2, W2, theta2):
    return _collision(np.array([x1, y1]), theta1, L1, W1, np.array([x2, y2]), theta2, L2, W2)

def plot(x1, y1, x2, y2, L1, W1, theta1, L2, W2, theta2):
    Acorners = _corner_points(np.array([x1, y1]), theta1, L1, W1)
    Bcorners = _corner_points(np.array([x2, y2]), theta2, L2, W2)
    Ax = [x for x, _ in Acorners] + [Acorners[0][0]]
    Ay = [y for _, y in Acorners] + [Acorners[0][1]]
    Bx = [x for x, _ in Bcorners] + [Bcorners[0][0]]
    By = [y for _, y in Bcorners] + [Bcorners[0][1]]

    plt.plot(Ax, Ay, Bx, By, marker='o')
    plt.title('blue is vehicle 1 and orange is vehicle 2')
    plt.axis('equal')
    plt.show()
    # blue is vehicle A and orange is vehicle B
