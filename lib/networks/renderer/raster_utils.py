import imageio
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import sys
import math
import trimesh
import os
import pdb
import imageio

def compute_near_far(vertices, extr, device='cuda'):
    '''
    vertices: [10475, 3]: tensor of 3D points in world space 
    extr: [3, 4]: camera extrinsic matrix
    '''

    extr_4x4 = torch.cat([extr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)], dim=0).to(device)

    correction_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    extr_opengl = torch.matmul(correction_matrix, extr_4x4)

    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)
    vertices_h = torch.cat([vertices, ones], dim=1).to(device)

    vertices_cam = torch.matmul(vertices_h, extr_opengl.T)

    z_vals = vertices_cam[:, 2]

    near = torch.max(z_vals)
    far = torch.min(z_vals)

    return near, far

def perspective_projection_opencv_to_opengl(fx, fy, cx, cy, near, far, width, height, device='cuda'):

    fovy_rad = compute_fov_from_focal_length(fy, height)
    fovx_rad = compute_fov_from_focal_length(fx, width)

    near = -near
    far = -far

    cot_half_fov = 1 / math.tan(fovy_rad / 2)

    aspect = width / height

    P_clip = torch.tensor([
        [1 / (aspect * math.tan(fovy_rad / 2)), 0, 0, 0],
        [0, 1 / math.tan(fovy_rad / 2), 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=torch.float32, device=device)

    c_x = cx
    c_y_flipped = height - cy

    p_x = c_x - width / 2
    p_y = c_y_flipped - height / 2

    # Handle scaling, if necessary
    scale = 0.001
    scale *= 2.0

    translate = torch.tensor([
        [1, 0, 0, -scale * p_x],
        [0, 1, 0, scale * p_y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    return P_clip, translate

def world_to_clip_space(vertices, extr, P_clip, translate, device='cuda'):

    extr_4x4 = torch.cat([extr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)], dim=0)

    correction_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    extr_corrected = torch.matmul(correction_matrix, extr_4x4)

    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)
    vertices_h = torch.cat([vertices, ones], dim=1)

    vertices_cam = torch.matmul(vertices_h, extr_corrected.T)

    mtx = torch.matmul(P_clip, translate)

    vertices_clip = torch.matmul(vertices_cam, mtx.T)

    vertices_clip = vertices_clip / vertices_clip[:, 3:4]

    return vertices_clip[:, :4]


def compute_fov_from_focal_length(focal_length, image_size):

    return 2 * math.atan(image_size / (2 * focal_length))


def find_3d_vertices_for_uv_torch_dict(faces: torch.Tensor) -> dict:
    """
    Args:
        faces: [F, 3, 2] tensor, (vertex_idx, uv_idx)
    Returns:
        uv_to_vertices: Dict[int, Set[int]] where key = uv_idx, value = set of vertex_idx
    """
    uv_to_vertices = {}

    flat = faces.view(-1, 2)  # [F*3, 2]
    vert_idx = flat[:, 0].tolist()
    uv_idx = flat[:, 1].tolist()

    for v_idx, u_idx in zip(vert_idx, uv_idx):
        if u_idx not in uv_to_vertices:
            uv_to_vertices[u_idx] = set()
        uv_to_vertices[u_idx].add(v_idx)

    return uv_to_vertices


def load_obj(file_path):

    vertices = []
    faces = []
    uvs = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == 'v':

                x, y, z = map(float, tokens[1:4])
                vertices.append((x, y, z))

            elif tokens[0] == 'vt':

                u, v = map(float, tokens[1:3])
                uvs.append((u, v))

            elif tokens[0] == 'f':

                face = []
                for token in tokens[1:]:
                    vertex_info = token.split('/')
                    vertex_index = int(vertex_info[0]) - 1
                    uv_index = int(vertex_info[1]) - 1 if len(
                        vertex_info) > 1 else None
                    face.append((vertex_index, uv_index))
                faces.append(face)

    return vertices, faces, uvs
