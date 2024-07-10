import math
import numpy as np
from typing import NamedTuple
import torch


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale = 1.0):
    # 这个函数适用于需要将世界坐标系的点转换到相机视图坐标系的场景，考虑了旋转、平移、额外的平移以及缩放因素。
    Rt = np.zeros((4,4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    C2W = np.linalg.inv(Rt)  # 计算 Rt 的逆矩阵 C2W，表示从相机坐标系到世界坐标系的变换。
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate)*scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
    
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovX = math.tan((fovX / 2))
    tanHalfFovY = math.tan((fovY / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    
    P = torch.zeros(4, 4)
    
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))