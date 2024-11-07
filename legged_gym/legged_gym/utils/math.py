import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:, :2] = 0.0
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw

def in_poly_2d(corners: torch.Tensor, point: torch.Tensor, margin: float=0) -> torch.Tensor:
    """input: corners in order (n_env,k,2), point (n_env,2), margin some float >=0;
       output: booleans (n_env,) on whether the point is inside each 4 corners
       currently support only convex polygon"""
    assert margin >= 0
    results = torch.ones(point.shape[0], dtype=torch.bool, device = point.device)
    point2corners = corners - point.unsqueeze(1) # n,k,2
    # check inside: point2corners[:,edge] cross point2corners[:,edge+1] should be with the same sign
    # check margin: d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    for edge in range(-2, corners.shape[1]-2): 
        next_edge = edge + 1
        # point2corners[:,edge] cross point2corners[:,edge+1] should be with the same sign
        cross_1 = point2corners[:,edge,0] * point2corners[:,edge+1,1] - point2corners[:,edge+1,0] * point2corners[:,edge,1]
        cross_2 = point2corners[:,edge+1,0] * point2corners[:,edge+2,1] - point2corners[:,edge+2,0] * point2corners[:,edge+1,1]
        same_sign = (cross_1 * cross_2 >= 0)
        results = torch.logical_and(results, same_sign)
        dist = torch.abs(cross_1) / torch.norm(corners[:,edge+1]-corners[:,edge], dim=-1)
        results = torch.logical_and(results, dist>=margin) 
    return results
    # # test example ------------------------------------------------------
    # corners = torch.Tensor([[0,0],[0,1],[0,1.3],[1,0.9],[2,0]]).unsqueeze(0)
    # point = torch.Tensor([0.5,0.5]).unsqueeze(0)
    # print(in_poly_2d(corners, point,0))  # tensor([True])
    # print(in_poly_2d(corners, point,0.3))  # tensor([True])
    # print(in_poly_2d(corners, point,0.5))  # tensor([True])
    # print(in_poly_2d(corners, point,0.8))  # tensor([False])
    # # -----------------------------------------------------------------------

def get_height_in_plane(triangle: np.ndarray, p_2d: np.ndarray) -> np.float32:
    p1, p2, p3 = triangle
    x, y = p_2d
    vec1 = p2 - p1
    vec2 = p3 - p2 
    normal1_vec = np.cross(vec1, vec2)
    A, B, C = normal1_vec
    D = -np.dot(normal1_vec, p1)
    z = -(A*x + B*y + D) / C 
    return z

def batch_rand_int(upper_bound: torch.Tensor):
    upper_bound_float =upper_bound.float()
    rand_floats = torch.rand_like(upper_bound_float)
    sampled_ints = (rand_floats * upper_bound_float).long()
    return sampled_ints


def diff_quat(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the difference between two quaternions.
    Quaternions are in the format [x, y, z, w].

    Parameters:
    - quat1: A tensor of shape (num_env, 4) representing the first quaternion(s).
    - quat2: A tensor of shape (num_env, 4) representing the second quaternion(s).

    Returns:
    - A tensor of shape (num_env, 4) representing the quaternion difference.
    """
    # Ensure quaternion inputs are of correct shape and type
    assert quat1.shape[-1] == 4 and quat2.shape[-1] == 4, "Quaternions must be of shape (*, 4)"

    # Calculate the conjugate of quat1
    quat1_conj = quat1.clone()
    quat1_conj[:, :3] = -quat1_conj[:, :3]  # Negate the x, y, z components to get the conjugate

    # Calculate the norm squared of quat1
    norm_sq_quat1 = torch.sum(quat1 * quat1, dim=1, keepdim=True)

    # Calculate the inverse of quat1
    quat1_inv = quat1_conj / norm_sq_quat1

    # Calculate the difference as quat2 * quat1_inv
    qd = quaternion_multiply(quat2, quat1_inv)

    # Normalize the quaternion difference to ensure w is within [-1, 1]
    qd_norm = torch.linalg.norm(qd, dim=1, keepdim=True)
    qd_normalized = qd / qd_norm

    # Calculate the angle of rotation from the quaternion difference
    angle = 2 * torch.acos(torch.clamp(qd_normalized[:, 3], -1.0, 1.0))

    return angle

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Quaternions are in the format [x, y, z, w].

    Parameters:
    - q1, q2: Tensors of shape (num_env, 4).

    Returns:
    - The product of the two quaternions, shape (num_env, 4).
    """
    # Extract components for readability
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Calculate the components of the product
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    # Combine the components into a single tensor
    return torch.stack((x, y, z, w), dim=-1)
