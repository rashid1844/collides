import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.stats import chi2

@njit
def _rotation_matrix(theta):
    """2d Rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]],np.float64)
    return R


@njit
def _corner_points(center, theta, L, W):
    """finds four corners of vehicle, based on center, theta, L, W"""
    L_vec = np.array([L / 2 * np.cos(theta), L / 2 * np.sin(theta)], np.float64)
    W_vec = np.array([-W / 2 * np.sin(theta), W / 2 * np.cos(theta)], np.float64)
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


@njit
def _approx(mean1, mean2, cov1, cov2):
    xi = np.array([[(mean1[0]-mean2[0])], [(mean1[1]-mean2[1])]])
    covi = cov1+cov2
    x_unit = xi /np.linalg.norm(xi)
    var1 = np.sum(x_unit.T@covi@x_unit)
    return np.linalg.norm(xi), np.sqrt(var1)


@njit
def _safe_distance(center1, theta1, L1, W1, center2, theta2, L2, W2):
    dist = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
    Acorners = _corner_points(center1, theta1, L1, W1)
    Bcorners = _corner_points(center2, theta2, L2, W2)

    min_dist = W1 / 2 + W2 / 2
    max_dist = np.sqrt((L1 / 2) ** 2 + (W1 / 2) ** 2) + np.sqrt((L2 / 2) ** 2 + (W2 / 2) ** 2)

    safe_dist = max(_safe_dist_ref(center1, theta1, L1, W1, Bcorners, center2, min_dist, max_dist, dist),
                    _safe_dist_ref(center2, theta2, L2, W2, Acorners, center1, min_dist, max_dist, dist))

    return safe_dist


@njit
def _collision_transformed(corners, ref_L, ref_W):
    """safe bool for trasformed corners"""
    for c in corners:
        if -ref_L/2<=c[0]<=ref_L/2 and -ref_W/2<=c[1]<=ref_W/2:
            return True
    return False

@njit
def _safe_dist_ref(ref_center, ref_theta, ref_L, ref_W, corners, center, min_dist, max_dist, dist):
    """moves dist to find safe dist"""
    R = _rotation_matrix(-ref_theta)
    c1, c2, c3, c4, center = R @ (corners[0] - ref_center), R @ (corners[1] - ref_center), R @ (corners[2] - ref_center), R @ (corners[3] - ref_center), R @ (center - ref_center)

    dist_theta = np.arctan(center[1] / center[0])
    dist_x_y = np.array([abs(np.cos(dist_theta)), abs(np.sin(dist_theta))])
    dist_x_y[0] *= -1 if center[0] > 0 else 1  # increase or decrease x val depending on value
    dist_x_y[1] *= -1 if center[1] > 0 else 1
    l = float(min_dist)
    h = float(max_dist) + 1
    mid = l + (h - l) / 2
    while h - l > 0.1:
        mid = l + (h - l) / 2
        temp_corners = [c1 + (dist - mid) * dist_x_y, c2 + (dist - mid) * dist_x_y, c3 + (dist - mid) * dist_x_y, c4 + (dist - mid) * dist_x_y]
        safe = _collision_transformed(temp_corners, ref_L, ref_W)
        if safe:
            l = mid
        else:
            h = mid

    return mid


def _epsilon_shadow(center1, cov1, theta1, L1, W1, center2, cov2, theta2, L2, W2):
    if theta1 > np.pi / 2:
        theta1 -= np.pi
    elif theta1 < np.pi / 2:
        theta1 += np.pi

    if theta2 > np.pi / 2:
        theta2 -= np.pi
    elif theta2 < np.pi / 2:
        theta2 += np.pi

    cov1_inv = np.linalg.inv(cov1)
    cov2_inv = np.linalg.inv(cov2)
    A_var = np.array([cov1_inv[0][0], 2 * cov1_inv[0][1], cov1_inv[1][1]])
    B_var = np.array([cov2_inv[0][0], 2 * cov2_inv[0][1], cov2_inv[1][1]])

    low = float(0)
    high = float(1)
    while high - low > 0.0001:
        mid = low + (high - low) / 2
        chi2_inv = chi2.ppf(1 - mid, 1)
        safe = not _epsilon_collision(center1, cov1, theta1, L1, W1, center2, cov2, theta2, L2, W2, mid, A_var, B_var, chi2_inv)
        if safe:
            high = mid
        else:
            low = mid

    return mid


@njit
def _epsilon_collision(center1, cov1, theta1, L1, W1, center2, cov2, theta2, L2, W2, epsilon, A_var, B_var, chi2_inv):

    A1, B1, C1 = A_var / chi2_inv
    A2, B2, C2 = B_var / chi2_inv

    L1x1 = np.sqrt( 1 / (A1 + np.tan(theta1)**2 *C1 + np.tan(theta1) *B1) )
    #L1x2 = np.tan(theta1) * L1x1

    if theta1 > 0:
        theta1w = theta1 - np.pi / 2
    else:
        theta1w = theta1 + np.pi / 2
    W1x1 = np.sqrt(1 / (A1 + np.tan(theta1w) ** 2 * C1 + np.tan(theta1w) * B1))
    #W1x2 = np.tan(theta1w) * W1x1

    L2x1 = np.sqrt(1 / (A2 + np.tan(theta2) ** 2 * C2 + np.tan(theta2) * B2))
    L2x2 = np.tan(theta2) * L2x1

    if theta2 > 0:
        theta2w = theta2 - np.pi / 2
    else:
        theta2w = theta2 + np.pi / 2
    W2x1 = np.sqrt(1 / (A2 + np.tan(theta2w) ** 2 * C2 + np.tan(theta2w) * B2))
    W2x2 = np.tan(theta2w) * W2x1

    new_L1 = L1 + 2 * np.sqrt(L1x1 ** 2 + L2x2 ** 2)
    new_W1 = W1 + 2 * np.sqrt(W1x1 ** 2 + W2x2 ** 2)
    new_L2 = L2 + 2 * np.sqrt(L2x2 ** 2 + L2x2 ** 2)
    new_W2 = W2 + 2 * np.sqrt(W2x2 ** 2 + W2x2 ** 2)

    collide = _collision(center1, theta1, new_L1, new_W1, center2, theta2, new_L2, new_W2)

    return collide


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


def collides(x1, y1, x2, y2, L1, W1, theta1, L2, W2, theta2, cov1=None, cov2=None, method=None, sample_size=100):
    if cov1 is None or cov2 is None or method is None:
        return _collision(np.array([x1, y1],np.float64), theta1, L1, W1, np.array([x2, y2],np.float64), theta2, L2, W2)

    if method == 's':
        sx1, sy1 = np.random.multivariate_normal([x1, y1], cov1, sample_size).T
        sx2, sy2 = np.random.multivariate_normal([x2, y2], cov2, sample_size).T
        return sum(1 for i in range(sample_size) if _collision(np.array([sx1[i], sy1[i]]), theta1, L1, W1, np.array([sx2[i], sy2[i]]), theta2, L2, W2)) / sample_size

    elif method == 'g':
        safe_dist = _safe_distance(np.array([x1, y1],np.float64), theta1, L1, W1, np.array([x2, y2],np.float64), theta2, L2, W2)
        mu, var = _approx(np.array([x1, y1],np.float64), np.array([x2, y2],np.float64), np.asarray(cov1,np.float64), np.asarray(cov2,np.float64))
        return ndtr((safe_dist - mu) / var)

    elif method == 'e':
        return 1 - (1 - _epsilon_shadow(np.array([x1, y1],np.float64), np.asarray(cov1,np.float64), theta1, L1, W1, np.array([x2, y2],np.float64), np.asarray(cov2,np.float64),theta2, L2, W2)) ** 2

    return None