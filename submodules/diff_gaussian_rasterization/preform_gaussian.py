import torch
import numpy as np
import math
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh
   

# def in_frustum(points, viewmatrix, projmatrix, prefiltered):
#     P = points.size(0)
#     p_orig_hom = torch.cat(points, torch.ones((P, 1)))
#     p_hom = projmatrix @ p_orig_hom.T
#     p_w = 1.0 / (p_hom[:, 3] + 0.0000001)
#     p_proj = [p_orig_hom[:, 0] * p_w, p_orig_hom[:, 1] * p_w, p_orig_hom[:, 2] * p_w]
#     p_view = viewmatrix @ p_orig_hom.T
    
#     mask = (p_view[:, 2] <= 0.2 and not prefiltered).squeeze()
#     return mask
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
    t = p_orig_hom @ viewmatrix
    
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[:, 0] / t[:, 2]
    tytz = t[:, 1] / t[:, 2]
    # t[:, 0] = min(limx, max(-limx, txtz)) * t[:, 2]
    # t[:, 1] = min(limy, max(-limx, tytz)) * t[:, 2]
    t[:, 0] = torch.min(torch.tensor(limx, device='cuda'), torch.max(torch.tensor(-limx, device='cuda'), txtz)) * t[:, 2]
    t[:, 1] = torch.min(torch.tensor(limy, device='cuda'), torch.max(torch.tensor(-limy, device='cuda'), tytz)) * t[:, 2]
    
    ## 计算雅可比矩阵J
    J = torch.zeros((P, 3, 3), dtype=torch.float, device='cuda')
    J[:, 0, 0] = fovx / t[:, 2]
    J[:, 0, 2] = -(fovx * t[:, 0]) / (t[:, 2] * t[:, 2])
    J[:, 1, 1] = fovy / t[:, 2]
    J[:, 1, 2] = -(fovy * t[:, 1]) / (t[:, 2] * t[:, 2])
    
    ## 计算相机旋转矩阵W
    W = viewmatrix[:3, :3].T
    W_expanded = W.unsqueeze(0).expand(P, -1, -1)
    
    M = torch.bmm(J, W_expanded)
    
    Vrk = torch.zeros((P, 3, 3), dtype=torch.float, device='cuda')
    Vrk[:, 0, 0] = covariance3D[:, 0]
    Vrk[:, 0, 1] = Vrk[:, 1, 0] = covariance3D[:, 1]
    Vrk[:, 0, 2] = Vrk[:, 2, 0] = covariance3D[:, 1]
    Vrk[:, 1, 1] = covariance3D[:, 3]
    Vrk[:, 1, 2] = Vrk[:, 2, 1] = covariance3D[:, 4]
    Vrk[:, 2, 2] = covariance3D[:, 5]
    
    cov = torch.bmm(torch.bmm(M, Vrk), M.transpose(1, 2))
    cov2D = torch.zeros((P, 3), dtype=float, device='cuda')
    cov2D[:, 0] = cov[:, 0, 0] + 0.3
    cov2D[:, 1] = cov[:, 0, 1]
    cov2D[:, 2] = cov[:, 1, 1] + 0.3
    
    return cov2D  
 
def ndc2Pix(f, i):
    return ((f + 1.0) * i - 1) * 0.5
    
def getRect(point_2D, max_radius, width, height):    
    num = point_2D.shape[0]
    grid_x = (width + TILE_X - 1) // TILE_X
    grid_y = (height + TILE_Y - 1) // TILE_Y

    rect_min = torch.zeros((num, 2), dtype=torch.uint8, device='cuda')
    rect_max = torch.zeros((num, 2), dtype=torch.uint8, device='cuda')
    
    rect_min[:, 0] = torch.min(torch.tensor(grid_x, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 0] - max_radius[:] + TILE_X - 1) // TILE_X))
    rect_min[:, 1] = torch.min(torch.tensor(grid_y, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 1] - max_radius[:] + TILE_Y - 1) // TILE_Y))
    
    rect_max[:, 0] = torch.min(torch.tensor(grid_x, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 0] + max_radius[:] + TILE_X - 1) // TILE_X))
    rect_max[:, 1] = torch.min(torch.tensor(grid_y, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 1] + max_radius[:] + TILE_Y - 1) // TILE_Y))
    return rect_min, rect_max

def getPixRect(point_2D, max_radius, width, height):
    num = point_2D.shape[0]
    pix_min = torch.zeros((num, 2), dtype=torch.uint8, device='cuda')
    pix_max = torch.zeros((num, 2), dtype=torch.uint8, device='cuda')
    
    pix_min[:, 0] = torch.min(torch.tensor(width, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 0] - max_radius[:] + width - 1) // width))
    pix_min[:, 1] = torch.min(torch.tensor(height, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 1] - max_radius[:] + height - 1) // height))
    
    pix_max[:, 0] = torch.min(torch.tensor(width, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 0] + max_radius[:] + width - 1) // width))
    pix_max[:, 1] = torch.min(torch.tensor(height, device='cuda'), torch.max(torch.tensor(0, device='cuda'), (point_2D[:, 1] + max_radius[:] + height - 1) // height))
    
    return pix_min, pix_max

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
     
def preprocess(P, D, M, points, scales, scale_modifier, rotations, opacities, shs, cov3D_precomp, colors_precomp, 
               viewmatrix, projmatrix, cam_pos, W, H, tan_fovx, tan_fovy, focal_x, focal_y, prefiltered):
    data_device = points.device
    p_one = torch.ones((P, 1), device=data_device)
    p_orig_hom = torch.cat((points, p_one), dim=1)
    p_view =  p_orig_hom @ viewmatrix
    in_frustum_mask = (p_view[:, 2] >= 0.2).squeeze()  # 掩码1：判断是否在是椎体内的mask
    
    points_in_frustum = points[in_frustum_mask]  # 在视锥体内的高斯
    P_in_frustum = points_in_frustum.shape[0]
    
    p_orig_hom_in_frustum = p_orig_hom[in_frustum_mask]
    p_hom = p_orig_hom_in_frustum @ projmatrix
    p_w = 1.0 / (p_hom[:, 3] + 0.0000001)
    
    p_proj = torch.zeros((P_in_frustum, 3), dtype=float, device=data_device)
    # p_proj[:,] = [p_orig_hom_in_frustum[:, 0] * p_w[:], p_orig_hom_in_frustum[:, 1] * p_w[:], p_orig_hom_in_frustum[:, 2] * p_w[:]]
    # p_proj[:, 0] = p_hom[:, 0] * p_w[:]
    # p_proj[:, 1] = p_hom[:, 1] * p_w[:]
    # p_proj[:, 2] = p_hom[:, 2] * p_w[:]
    p_proj = p_hom * p_w[:, None]
    
    ## 计算3D协方差
    if cov3D_precomp.shape[0] != 0:
        cov3D = cov3D_precomp[in_frustum_mask]
    else:
        L = build_scaling_rotation(scale_modifier * scales[in_frustum_mask], rotations[in_frustum_mask])
        actral_covariance = L @ L.transpose(1,2)
        cov3D = strip_symmetric(actral_covariance)
       
    ## 计算2D协方差 
    cov2D = computeCov2D(points_in_frustum, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix)
    
    det = torch.zeros((P_in_frustum), dtype=torch.float, device=data_device)
    det[:] = cov2D[:, 0] * cov2D[:, 2] - cov2D[:, 1] * cov2D[:, 1]
    if_inv_mask = (det[:] != 0)  # 掩码2：判断二维协方差是否可逆的mask
    
    points_in_frustum_and_inv = points_in_frustum[if_inv_mask]  # 在视锥体内且二维协方差可逆的高斯
    P_in_frustum_and_inv = points_in_frustum_and_inv.shape[0]
    det_not0 = det[if_inv_mask]
    cov2D_det_not0 = cov2D[if_inv_mask]
    
    ## 求二维协方差的逆
    det_inv = torch.zeros((P_in_frustum_and_inv), dtype=torch.float, device=data_device)
    det_inv[:] = 1 / det_not0[:]
    conic = torch.zeros((P_in_frustum_and_inv, 3), dtype=torch.float, device=data_device)
    conic[:, 0] = cov2D_det_not0[:, 2] * det_inv[:]
    conic[:, 1] = -cov2D_det_not0[:, 1] * det_inv[:]
    conic[:, 2] = cov2D_det_not0[:, 0] * det_inv[:]
    
    ## 求二维协方差的特征值
    mid = torch.zeros((P_in_frustum_and_inv), dtype=float, device=data_device)
    lambda1 = torch.zeros((P_in_frustum_and_inv), dtype=float, device=data_device)
    lambda2 = torch.zeros((P_in_frustum_and_inv), dtype=float, device=data_device)
    
    mid[:] = 0.5 * (cov2D_det_not0[:, 0] + cov2D_det_not0[:, 2])
    lambda1[:] = mid[:] + torch.sqrt(torch.max(torch.tensor(0.1, device=data_device), mid[:] * mid[:] - det_not0[:]))
    lambda2[:] = mid[:] - torch.sqrt(torch.max(torch.tensor(0.1, device=data_device), mid[:] * mid[:] - det_not0[:]))
    
    ## 求最长的半轴
    my_radius = torch.zeros((P_in_frustum_and_inv), dtype=float, device=data_device)
    my_radius[:] = torch.ceil(3.0 * torch.sqrt(lambda1[:]))
    
    ## 将ndc坐标转换为像素坐标
    point_image = torch.zeros((P_in_frustum_and_inv, 2), dtype=torch.float, device=data_device)
    point_image[:, 0] = ndc2Pix(p_proj[if_inv_mask][:, 0], W)
    point_image[:, 1] = ndc2Pix(p_proj[if_inv_mask][:, 1], H)
    
    ## 筛选高斯是否对tile有贡献
    rect_min, rect_max = getRect(point_image, my_radius, W, H)
    tile_touched0_mask = ((rect_max[:, 0] - rect_min[:, 0]) * (rect_max[:, 1] - rect_min[:, 1]) != 0).squeeze()  # 掩码3：有贡献高斯掩码
    points_attribute = points_in_frustum_and_inv[tile_touched0_mask]
    P_attribute = points_attribute.shape[0]
    
    radii = torch.zeros(P_attribute, dtype=torch.int32, device=data_device)
    tiles_touched = torch.zeros(P_attribute, dtype=torch.int32, device=data_device)
    depths = torch.zeros(P_attribute, dtype=torch.float32, device=data_device)
    points_xy_image = torch.zeros((P_attribute, 2), dtype=torch.float32, device=data_device)
    rgb = torch.zeros((P_attribute, 3), dtype=torch.float32, device=data_device)
    conic_opacity = torch.zeros((P_attribute, 4), dtype=torch.float32, device=data_device)
    
    ## 计算颜色
    shs_attribute = shs[in_frustum_mask][if_inv_mask][tile_touched0_mask]
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
    radii = my_radius[tile_touched0_mask]
    tiles_touched = (rect_max[tile_touched0_mask][..., 0] - rect_min[tile_touched0_mask][..., 0]) * (rect_max[tile_touched0_mask][..., 1] - rect_min[tile_touched0_mask][..., 1])
    depths = p_view[in_frustum_mask][if_inv_mask][tile_touched0_mask][..., 2]
    points_xy_image = point_image[tile_touched0_mask]
    conic_opacity = torch.cat((conic[tile_touched0_mask], opacities[in_frustum_mask][if_inv_mask][tile_touched0_mask].unsqueeze(1)), dim=1)
    
    return radii, tiles_touched, depths, points_xy_image, rgb, conic_opacity, rect_min[tile_touched0_mask], rect_max[tile_touched0_mask]
    
def render_per_pixel(points_xy_image, tile_gs_pair, rgb, conic_opacity, radii, tiles_x, tiles_y, W, H):
    # tile_idx_gs = torch.cat((tile_gs_pair, torch.arange(0, tiles_x * tiles_y).unsqueeze(1).to(points_xy_image.device)), dim=1)
    
    # dim = 1 的地方，第一个数放像素索引，第二个数放像素对应的tile索引，第三个元素放计算的颜色
    rgb_for_pixel = torch.cat((torch.arange(0, W * H).unsqueeze(1), torch.zeros((W * H, 2))), dim=1)
    # 计算每个像素对应的tile
    rgb_for_pixel[:, 1] = rgb_for_pixel[:, 0] // W // TILE_X * tiles_x + rgb_for_pixel[:, 0] % W // TILE_Y
    
    pix_min, pix_max = getPixRect(points_xy_image, radii, W, H)
    T = torch.ones(W * H, dtype=torch.float32, device=points_xy_image.device)
    
    for gs in range(points_xy_image.shape[0]):
        gs_idx = tile_gs_pair[rgb_for_pixel[:, 1], gs]
        
        
    return
    
    
        
    
    
    
    
    
        
    
    