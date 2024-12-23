# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 12/23/24 1:49 PM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : lbs.py.py
# @desc     :

import numpy as np
import torch
import torch.nn.functional as F
from numpy.ma.core import zeros
from zmq.backend.cffi import device

from tutorials.ik_example_joints import batch_size


def to_tensor(array, dtype=torch.float32)-> 'torch.Tensor':
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

class Struct(object):
    """
    A general structure to hold variables
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def to_np(array, dtype=np.float32)-> 'np.array':
    """
    Convert a torch.Tensor to np.array, if the input is a scipy.sparse,
    it will be converted to dense matrix first.
    Parameters
    ----------
    array: Tensor array
    dtype: default is np.float32

    Returns: np.array
    -------

    """
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for the extreme cases of eular angles like [0.0, pi, 0.0]
    sy = torch.sqrt(
        rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
        rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
    )
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def vertices2joints(J_regressor, vertices):
    """
    Calculate the 3D joint locations from the vertices of the mesh

    Parameters
    ----------
    J_regressor: torch.tensor JxV
        The regressor array that is used to calculate the joints from
        the position of the vertices
    vertices: torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch. tensor BxJx3
        The location of the joints

    """
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

def blend_shapes(betas, shape_disps):
    """
    Calculates the per vertex displacement due to the blend shapes
    Parameters
    ----------
    betas: torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per vertex displacement due to the blend shapes
    """
    blend_shape = torch.einsum('bl, mkl->bmk', [betas, shape_disps])
    return blend_shape

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """
    Calculate the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors

    Returns
    -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given rotation vectors
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
        dim=1
    ).view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def transform_mat(R, t):
    """
    Create a batch of transformation matrices
    Parameters
    ----------
    R: Bx3x3 array of a batch of rotation matrices
    t: Bx3x1 array of a batch of translation vectors

    Returns
    -------
    Rt: Bx4x4 Transforamtion matrix
    """
    return torch.cat([F.pad(R, (0, 0, 0, 1)),
                      F.pad(t, (0, 0, 0, 1), value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats: torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints: torch.tensor BxNx3
        Locations of the joints
    parents: torch.tensor BxN
        The kinematic tree of each object

    Returns
    -------
    posed_joints: torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms: torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # substract the joint location at the rest pose
        # no need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]
    # question: why original code pad with 0?
    joints_homogen = F.pad(joints, (0, 0, 0 ,1), value=1)

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), (3, 0, 0, 0, 0, 0, 0, 0)
    )
    return posed_joints, rel_transforms


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, joints=None, pose2rot=True, v_shaped=None, dtype=torch.float32):
    """
    Perform Linear Blend Skinning with the given shape and pose parameters
    Parameters
    ----------
    betas: torch.tensor BxNB
        The tensor of shape parameters
    pose: torch.tensor Bx(J + 1) * 3
        The tensor of pose parameters in axis-angle format
    v_template: torch.tensor BxVx3
        The template mesh that will be deformed
    shapedirs: torch.tensor 1xNB
        The tensor of PCA shape displacements
    posedirs: torch.tensor Px(V * 3)
        The tensor of pose PCA coefficients
    J_regressor: torch.tensor JxV
        The regressor array that is used to calculate the joints
        form the position of the vertices
    parents: torch.tensor J
        The array that describes the kinematic tree of the model
    lbs_weights: torch.tensor N x V x (J + 1)
        The linear blend skinning weights that represent how much the
        rotation matrix of each part affects each vertex
    pose2rot: bool, optional
        Flag on whether to convert the input pose tensor to rotation matrices.
        The default value is True. If False, then the pose tensor should already
        contain rotation matrices and have a size of Bx(J+1)x9

    Returns
    -------
    verts: torch.tensor BxVx3
        The vertices of the mesh after applying the shape and pose displacements.
    joints: torch.tensor BxJx3
        The joints of the model
    """
    if betas.shape[0] != pose.shape[0]:
        raise ValueError('The batch size of betas and pose should be the same')
    batch_size = betas.shape[0]
    device = betas.device

    # Add shape contribution
    if v_shaped is None:
        v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    if joints is not None:
        J = joints
    else:
        J = vertices2joints(J_regressor, v_shaped)

    # Add pose blend shapes
    # NxJx3x3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype
        ).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1, :, :] - ident).view(batch_size, -1)
        # (N x P) x (P x V x 3) - > N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # Skinning
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed



