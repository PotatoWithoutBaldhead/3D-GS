import torch
import numpy as np
import math
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh
import torchvision.transforms as transforms_T
from PIL import Image
import time
import tqdm
import math
import matplotlib.pyplot as plt
   
TILE_X = 16
TILE_Y = 16
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]  

def computeCov2D(points, fovx, fovy, tan_fovx, tan_fovy, covariance3D, viewmatrix):
    P = points.shape[0]
    p_one = torch.ones((P, 1), device='cuda')
    p_orig_hom = torch.cat((points,p_one), dim=1)
    tx = (p_orig_hom[..., 0] / p_orig_hom[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * points[..., 2]
    ty = (p_orig_hom[..., 1] / p_orig_hom[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * points[..., 2]
    tz = p_orig_hom[..., 2]

    ## 计算雅可比矩阵J
    J = torch.zeros((P, 3, 3), dtype=torch.float, device='cuda')
    J[..., 0, 0] = 1 / tz * fovx
    J[..., 0, 2] = -tx / (tz * tz) * fovx
    J[..., 1, 1] = 1 / tz * fovy
    J[..., 1, 2] = -ty / (tz * tz) * fovy
    
    ## 计算相机旋转矩阵W
    W = viewmatrix[:3, :3].T
    W_expanded = W.unsqueeze(0).expand(P, -1, -1)
    
    M = torch.bmm(J, W_expanded)
    cov = torch.bmm(torch.bmm(M, covariance3D), M.transpose(1, 2))
    cov[:, 0, 0] += 0.3
    cov[:, 1, 1] += 0.3
    
    return cov[:, 0:2, 0:2]
 
def ndc2Pix(f, i):
    return ((f + 1.0) * i - 1) * 0.5
    
def getRect(point_2D, max_radius, width, height):    
    num = point_2D.shape[0]
    grid_x = (width + TILE_X - 1) // TILE_X
    grid_y = (height + TILE_Y - 1) // TILE_Y

    rect_min = torch.zeros((num, 2), dtype=torch.uint8, device='cuda')
    rect_max = torch.zeros((num, 2), dtype=torch.uint8, device='cuda')
    
    rect_min[..., 0] = torch.clamp(((point_2D[..., 0] - max_radius[...]) // TILE_X), min=0, max=grid_x)
    rect_min[..., 1] = torch.clamp(((point_2D[..., 1] - max_radius[...]) // TILE_Y), min=0, max=grid_y)
    
    rect_max[..., 0] = torch.clamp(((point_2D[..., 0] + max_radius[...] + TILE_X - 1) // TILE_X), min=0, max=grid_x)
    rect_max[..., 1] = torch.clamp(((point_2D[..., 1] + max_radius[...] + TILE_Y - 1) // TILE_Y), min=0, max=grid_y)
    
    return rect_min, rect_max

def computeColorFromSH(deg, sh, dirs):
    assert deg <= 4 and deg >= 0
    result = C0 * sh[:, 0, :]
    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:]
        result = (result -
                  C1 * y * sh[:, 1, :] +
                  C1 * z * sh[:, 2, :] -
                  C1 * x * sh[:, 3, :])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[:, 4, :] +
                      C2[1] * yz * sh[:, 5, :] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[:, 6, :] +
                      C2[3] * xz * sh[:, 7, :] +
                      C2[4] * (xx - yy) * sh[:, 8, :])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[:, 9, :] +
                          C3[1] * xy * z * sh[:, 10, :] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[:, 11, :] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12, :] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[:, 13, :] +
                          C3[5] * z * (xx - yy) * sh[:, 14, :] +
                          C3[6] * x * (xx - 3 * yy) * sh[:, 15, :])

                if deg > 3:
                    result = (result + 
                              C4[0] * xy * (xx - yy) * sh[:, 16, :] +
                              C4[1] * yz * (3 * xx - yy) * sh[:, 17, :] +
                              C4[2] * xy * (7 * zz - 1) * sh[:, 18, :] +
                              C4[3] * yz * (7 * zz - 3) * sh[:, 19, :] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[:, 20, :] +
                              C4[5] * xz * (7 * zz - 3) * sh[:, 21, :] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[:, 22, :] +
                              C4[7] * xz * (xx - 3 * yy) * sh[:, 23, :] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[:, 24, :])
                    
    result += 0.5
    result_clamped = torch.clamp(result, min=0.0, max=1.0)
    return result_clamped
     
def preprocess(P, D, points, scales, scale_modifier, rotations, opacities, shs, cov3D_precomp, colors_precomp, 
               viewmatrix, projmatrix, cam_pos, W, H, tan_fovx, tan_fovy, focal_x, focal_y, prefiltered):
    data_device = points.device
    p_one = torch.ones((P, 1), device=data_device)
    p_orig_hom = torch.cat((points, p_one), dim=1)
    p_view =  p_orig_hom @ viewmatrix
    in_frustum_mask = (p_view[:, 2] > 0.2).squeeze()  # 掩码1：判断是否在是椎体内的mask
    
    points_in_frustum = points[in_frustum_mask]  # 在视锥体内的高斯
    
    p_orig_hom_in_frustum = p_orig_hom[in_frustum_mask]
    p_hom = p_orig_hom_in_frustum @ projmatrix
    p_w = 1.0 / (p_hom[:, 3] + 0.0000001)
    
    p_proj = p_hom * p_w[:, None]

    # else:
    L = build_scaling_rotation(scale_modifier * scales[in_frustum_mask], rotations[in_frustum_mask])
    cov3D = L @ L.transpose(1,2)
  
    ## 计算2D协方差 
    cov2D = computeCov2D(p_view[in_frustum_mask], focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix)

    det = torch.det(cov2D)
    if_inv_mask = (det != 0)  # 掩码2：判断二维协方差是否可逆的mask
    
    points_in_frustum_and_inv = points_in_frustum[if_inv_mask]  # 在视锥体内且二维协方差可逆的高斯
    P_in_frustum_and_inv = points_in_frustum_and_inv.shape[0]
    cov2D_det_not0 = cov2D[if_inv_mask]
    
    # 二维协方差的逆矩阵
    cov2D_inv = torch.inverse(cov2D_det_not0)
    conic = torch.zeros((P_in_frustum_and_inv, 3), dtype=torch.float, device=data_device)
    conic[:, 0] = cov2D_inv[:, 0, 0]
    conic[:, 1] = cov2D_inv[:, 0, 1]
    conic[:, 2] = cov2D_inv[:, 1, 1]
    
    eigenvalues = torch.linalg.eigvals(cov2D_det_not0)
    
    lambda1, _ = torch.max(eigenvalues.type(torch.float), dim=1, keepdim=True)
    my_radius = torch.ceil(3.0 * torch.sqrt(lambda1.squeeze(1)))
    
    ## 将ndc坐标转换为像素坐标
    point_image = torch.zeros((P_in_frustum_and_inv, 2), dtype=torch.float, device=data_device)
    point_image[:, 1] = ndc2Pix(p_proj[if_inv_mask][:, 1], H)
    
    ## 筛选高斯是否对tile有贡献
    rect_min, rect_max = getRect(point_image, my_radius, W, H)
    tile_touched_mask = ((rect_max[:, 0] - rect_min[:, 0]) * (rect_max[:, 1] - rect_min[:, 1]) != 0).squeeze()  # 掩码3：有贡献高斯掩码
    points_attribute = points_in_frustum_and_inv[tile_touched_mask]
    P_attribute = points_attribute.shape[0]
    
    depths = torch.zeros(P_attribute, dtype=torch.float32, device=data_device)
    rgb = torch.zeros((P_attribute, 3), dtype=torch.float32, device=data_device)
    conic_opacity = torch.zeros((P_attribute, 4), dtype=torch.float32, device=data_device)
    
    ## 计算颜色
    shs_attribute = shs[in_frustum_mask][if_inv_mask][tile_touched_mask]
    if colors_precomp.shape[0] == 0:
        dirs = torch.zeros((P_attribute, 3), dtype=torch.float, device=data_device)
        dirs[:, 0] = points_attribute[:, 0] - cam_pos[0]
        dirs[:, 1] = points_attribute[:, 1] - cam_pos[1]
        dirs[:, 2] = points_attribute[:, 2] - cam_pos[2]
        length_dir_inv = 1 / torch.sqrt(dirs[..., 0] ** 2 + dirs[..., 1] ** 2 + dirs[..., 2] ** 2)
        dirs = dirs * length_dir_inv[:, None]  # 方向单位化
        result = computeColorFromSH(D, shs_attribute, dirs)
        
        rgb[..., 0] = result[..., 0]
        rgb[..., 1] = result[..., 1]
        rgb[..., 2] = result[..., 2]
        
    ## 保存各个参数值
    depths = p_view[in_frustum_mask][if_inv_mask][tile_touched_mask][..., 2]
    points_xy_image = point_image[tile_touched_mask]
    conic_opacity = torch.cat((conic[tile_touched_mask], opacities[in_frustum_mask][if_inv_mask][tile_touched_mask]), dim=1)
    
    # 最终可视的高斯掩码
    visibility_filter = in_frustum_mask.clone()
    visibility_filter[in_frustum_mask] &= if_inv_mask
    visibility_filter[in_frustum_mask] &= tile_touched_mask
    
    return depths, points_xy_image, rgb, conic_opacity, rect_min[tile_touched_mask], rect_max[tile_touched_mask], my_radius[tile_touched_mask], visibility_filter
    
def IfTileInGS(idx, tile_x, rect_min, rect_max):
    idx_x = idx % tile_x
    idx_y = idx // tile_x
    X_in_gs = (idx_x >= rect_min[:, 0]) & (idx_x < rect_max[:, 0])
    Y_in_gs = (idx_y >= rect_min[:, 1]) & (idx_y < rect_max[:, 1])
    tile_in_gs = X_in_gs & Y_in_gs
    return tile_in_gs

def render_per_pixel(points_xy_image, rgb, conic_opacity, depths, background, tiles_x, tiles_y, W, H, rect_min, rect_max):
    
    output_color = torch.zeros((W, H, 3), dtype=torch.float, device='cuda')
    final_T = torch.zeros((W, H, 1), device=points_xy_image.device, dtype=torch.float) # 每一像素点的透明度

    for tile_idx in range(tiles_x * tiles_y):
        i = tile_idx % tiles_x
        j = tile_idx // tiles_x
        pixel_min_x = i * TILE_X
        pixel_min_y = j * TILE_Y
        pixel_max_x = min(W, pixel_min_x + TILE_X)
        pixel_max_y = min(H, pixel_min_y + TILE_Y)
        
        pix_num_x = min(TILE_X, pixel_max_x - pixel_min_x)
        pix_num_y = min(TILE_Y, pixel_max_y - pixel_min_y)
        
        if_idx_in_gs = IfTileInGS(tile_idx, tiles_x, rect_min, rect_max)
        filtered_gs_sorted = points_xy_image[if_idx_in_gs]

        gs_num_of_tile = filtered_gs_sorted.shape[0]
        
        # 初始化变量
        pixel_uv = torch.zeros((pix_num_x, pix_num_y, gs_num_of_tile, 2), device=points_xy_image.device, dtype=torch.float) # 像素网格坐标
        alpha = torch.zeros((pix_num_x, pix_num_y, gs_num_of_tile, 1), device=points_xy_image.device, dtype=torch.float) # 贡献度（不透明度）alpha

        # 维度对齐
        con_o = conic_opacity[if_idx_in_gs].unsqueeze(0).unsqueeze(0).expand(pix_num_x, pix_num_y, gs_num_of_tile, 4)
        rgb_c = rgb[if_idx_in_gs].unsqueeze(0).unsqueeze(0).expand(pix_num_x, pix_num_y, gs_num_of_tile, 3)
        # 赋值
        xy = filtered_gs_sorted.unsqueeze(0).unsqueeze(0).expand(pix_num_x, pix_num_y, gs_num_of_tile, 2)
        pixel_uv[..., 0] = (torch.arange(pix_num_x, device=points_xy_image.device, dtype=torch.float).unsqueeze(1).expand(pix_num_x, pix_num_y) + pixel_min_x).unsqueeze(2)
        pixel_uv[..., 1] = (torch.arange(pix_num_y, device=points_xy_image.device, dtype=torch.float).unsqueeze(0).expand(pix_num_x, pix_num_y) + pixel_min_y).unsqueeze(2)
        d = xy - pixel_uv
        
        power = (-0.5 * (con_o[..., 0] * d[..., 0] * d[..., 0] + con_o[..., 2] * d[..., 1] * d[..., 1]) - con_o[..., 1] * d[..., 0] * d[..., 1]).unsqueeze(-1)
        power = torch.where(power > 0, torch.tensor(-float('inf')), power)
        alpha[:, :, :, 0] = con_o[:, :, :, 3] * torch.exp(power[:, :, :, 0])

        T = torch.cumprod(1 - alpha, dim=2)
        T = T / (1 - alpha)
        
        conrtibute_value = alpha * T
        if T.shape[2] == 0:
            final_T[pixel_min_x:pixel_max_x, pixel_min_y:pixel_max_y] = 1
        else:
            final_T[pixel_min_x : pixel_max_x, pixel_min_y : pixel_max_y] = T[..., -1, 0:1] # 更新当前tile的final_T，final_T=π(i)(1-alpha)
        color = torch.sum(conrtibute_value * rgb_c, dim=2)
        output_color[pixel_min_x : pixel_max_x, pixel_min_y : pixel_max_y] = color
         
    output_color += final_T * background
    
    return output_color.permute(2, 0, 1)
            
            
    
    
    
    
    
        
    
    