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
    tx = (p_orig_hom[..., 0] / p_orig_hom[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * points[..., 2]
    ty = (p_orig_hom[..., 1] / p_orig_hom[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * points[..., 2]
    tz = p_orig_hom[..., 2]
    # t[:, 0] = min(limx, max(-limx, txtz)) * t[:, 2]
    # t[:, 1] = min(limy, max(-limx, tytz)) * t[:, 2]
    # t[:, 0] = torch.min(torch.tensor(limx, device='cuda'), torch.max(torch.tensor(-limx, device='cuda'), txtz)) * t[:, 2]
    # t[:, 1] = torch.min(torch.tensor(limy, device='cuda'), torch.max(torch.tensor(-limy, device='cuda'), tytz)) * t[:, 2]
    
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
    
    # Vrk = torch.zeros((P, 3, 3), dtype=torch.float, device='cuda')
    # Vrk[:, 0, 0] = covariance3D[:, 0]
    # Vrk[:, 0, 1] = Vrk[:, 1, 0] = covariance3D[:, 1]
    # Vrk[:, 0, 2] = Vrk[:, 2, 0] = covariance3D[:, 2]
    # Vrk[:, 1, 1] = covariance3D[:, 3]
    # Vrk[:, 1, 2] = Vrk[:, 2, 1] = covariance3D[:, 4]
    # Vrk[:, 2, 2] = covariance3D[:, 5]
    
    cov = torch.bmm(torch.bmm(M, covariance3D), M.transpose(1, 2))
    # cov2D = torch.zeros((P, 3), dtype=float, device='cuda')
    # cov2D[:, 0] = cov[:, 0, 0] + 0.3
    # cov2D[:, 1] = cov[:, 0, 1]
    # cov2D[:, 2] = cov[:, 1, 1] + 0.3
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
     
def preprocess(P, D, M, points, scales, scale_modifier, rotations, opacities, shs, cov3D_precomp, colors_precomp, 
               viewmatrix, projmatrix, cam_pos, W, H, tan_fovx, tan_fovy, focal_x, focal_y, prefiltered):
    data_device = points.device
    p_one = torch.ones((P, 1), device=data_device)
    p_orig_hom = torch.cat((points, p_one), dim=1)
    p_view =  p_orig_hom @ viewmatrix
    in_frustum_mask = (p_view[:, 2] > 0.2).squeeze()  # 掩码1：判断是否在是椎体内的mask
    
    points_in_frustum = points[in_frustum_mask]  # 在视锥体内的高斯
    P_in_frustum = points_in_frustum.shape[0]
    
    p_orig_hom_in_frustum = p_orig_hom[in_frustum_mask]
    p_hom = p_orig_hom_in_frustum @ projmatrix
    p_w = 1.0 / (p_hom[:, 3] + 0.0000001)
    
    p_proj = p_hom * p_w[:, None]
    
    ## 计算3D协方差
    # if cov3D_precomp.shape[0] != 0:
    #     cov3D = cov3D_precomp[in_frustum_mask]
    # else:
    L = build_scaling_rotation(scale_modifier * scales[in_frustum_mask], rotations[in_frustum_mask])
    cov3D = L @ L.transpose(1,2)
    # cov3D = strip_symmetric(actral_covariance)
       
    ## 计算2D协方差 
    cov2D = computeCov2D(p_view[in_frustum_mask], focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix)

    det = torch.det(cov2D)
    if_inv_mask = (det != 0)  # 掩码2：判断二维协方差是否可逆的mask
    
    points_in_frustum_and_inv = points_in_frustum[if_inv_mask]  # 在视锥体内且二维协方差可逆的高斯
    P_in_frustum_and_inv = points_in_frustum_and_inv.shape[0]
    det_not0 = det[if_inv_mask]
    cov2D_det_not0 = cov2D[if_inv_mask]
    
    # 二维协方差的逆矩阵
    cov2D_inv = torch.inverse(cov2D_det_not0)
    conic = torch.zeros((P_in_frustum_and_inv, 3), dtype=torch.float, device=data_device)
    conic[:, 0] = cov2D_inv[:, 0, 0]
    conic[:, 1] = cov2D_inv[:, 0, 1]
    conic[:, 2] = cov2D_inv[:, 1, 1]
    
    eigenvalues = torch.linalg.eigvals(cov2D_det_not0)
    
    lambda1, _ = torch.max(eigenvalues.type(torch.float), dim=1, keepdim=True)
    my_radius = 3.0 * torch.ceil(torch.sqrt(lambda1.squeeze(1)))
    
    ## 将ndc坐标转换为像素坐标
    point_image = torch.zeros((P_in_frustum_and_inv, 2), dtype=torch.float, device=data_device)
    point_image[:, 0] = ndc2Pix(p_proj[if_inv_mask][:, 0], W)
    point_image[:, 1] = ndc2Pix(p_proj[if_inv_mask][:, 1], H)
    
    ## 筛选高斯是否对tile有贡献
    rect_min, rect_max = getRect(point_image, my_radius, W, H)
    # tile_touched0_mask = ((rect_max[:, 0] - rect_min[:, 0]) * (rect_max[:, 1] - rect_min[:, 1]) != 0).squeeze()  # 掩码3：有贡献高斯掩码
    # points_attribute = points_in_frustum_and_inv
    P_attribute = points_in_frustum_and_inv.shape[0]
    
    depths = torch.zeros(P_attribute, dtype=torch.float32, device=data_device)
    points_xy_image = torch.zeros((P_attribute, 2), dtype=torch.float32, device=data_device)
    rgb = torch.zeros((P_attribute, 3), dtype=torch.float32, device=data_device)
    conic_opacity = torch.zeros((P_attribute, 4), dtype=torch.float32, device=data_device)
    
    ## 计算颜色
    shs_attribute = shs[in_frustum_mask][if_inv_mask]
    if colors_precomp.shape[0] == 0:
        dirs = torch.zeros((P_attribute, 3), dtype=torch.float, device=data_device)
        dirs[:, 0] = points_in_frustum_and_inv[:, 0] - cam_pos[0]
        dirs[:, 1] = points_in_frustum_and_inv[:, 1] - cam_pos[1]
        dirs[:, 2] = points_in_frustum_and_inv[:, 2] - cam_pos[2]
        length_dir_inv = 1 / torch.sqrt(dirs[..., 0] ** 2 + dirs[..., 1] ** 2 + dirs[..., 2] ** 2)
        dirs = dirs * length_dir_inv[:, None]  # 方向单位化
        result = computeColorFromSH(D, shs_attribute, dirs)
        
        rgb[..., 0] = result[..., 0]
        rgb[..., 1] = result[..., 1]
        rgb[..., 2] = result[..., 2]
        
    ## 保存各个参数值
    depths = p_view[in_frustum_mask][if_inv_mask][..., 2]
    points_xy_image = point_image
    conic_opacity = torch.cat((conic, opacities[in_frustum_mask][if_inv_mask].unsqueeze(1)), dim=1)
    
    return depths, points_xy_image, rgb, conic_opacity, rect_min, rect_max
    
def render_per_pixel1(points_xy_image, tile_gs_pair, rgb, conic_opacity, radii, tiles_x, tiles_y, W, H):
    # tile_idx_gs = torch.cat((tile_gs_pair, torch.arange(0, tiles_x * tiles_y).unsqueeze(1).to(points_xy_image.device)), dim=1)
    
    # # dim = 1 的地方，第一个数放像素索引，第二个数放像素对应的tile索引，第三个元素放计算的颜色
    # rgb_for_pixel = torch.cat((torch.arange(0, W * H).unsqueeze(1), torch.zeros((W * H, 2))), dim=1)
    # # 计算每个像素对应的tile
    # rgb_for_pixel[:, 1] = rgb_for_pixel[:, 0] // W // TILE_X * tiles_x + rgb_for_pixel[:, 0] % W // TILE_Y
    start_time = time.time()
    rgb_for_pixel = torch.zeros((W, H, 3), dtype=torch.float32, device=points_xy_image.device)
    contributor_gs_num = torch.zeros(W*H, dtype=torch.float32, device=points_xy_image.device)
    for tile_idx in range(tiles_x * tiles_y):
        # 计算tile的左上角坐标
        tile_x_pix = (tile_idx % tiles_x) * TILE_X
        tile_y_pix = (tile_idx // tiles_x) * TILE_Y
        
        # 提取属于该tile的gs的id
        gs_for_tile = tile_gs_pair[tile_idx, :]
        mask = gs_for_tile != -1
        filtered_gs = gs_for_tile[mask]
        
        # dim0存像素索引， dim1存像素x坐标， dim2存像素y坐标
        pixel_for_tile = torch.zeros((TILE_X * TILE_Y, 3), device=points_xy_image.device)
        # pixel_gs_pair = torch.zeros((TILE_X * TILE_Y, 3), device=points_xy_image.device)
        for i in range(TILE_X):
            for j in range(TILE_Y):
                pixel_idx = (tile_y_pix + j) * W + (tile_x_pix + i)
                pixel_for_tile[i * TILE_X + j, 0] = pixel_idx
                pixel_for_tile[i * TILE_X + j, 1] = pixel_idx % W
                pixel_for_tile[i * TILE_X + j, 2] = pixel_idx // W
                # T = 0
                # contributor = 0
                # for gs in filtered_gs:
                #     d = [points_xy_image[gs, 0] - pixel_idx % W, points_xy_image[gs, 1] - pixel_idx // W]
                #     power = -0.5 * (conic_opacity[gs, 0] * d[0] * d[0] + conic_opacity[gs, 2] * d[1] * d[1]) - conic_opacity[gs, 1] * d[0] * d[1]
                #     if power > 0: 
                #         continue
                #     alpha = conic_opacity[gs, 3] * torch.exp(power)
                #     test_T = T * (1 - alpha)
                #     if test_T < 0.0001:
                #         break
                #     rgb_for_pixel[pixel_idx % W, pixel_idx // W, 0] += rgb[gs, 0] * alpha * T
                #     rgb_for_pixel[pixel_idx % W, pixel_idx // W, 1] += rgb[gs, 1] * alpha * T
                #     rgb_for_pixel[pixel_idx % W, pixel_idx // W, 2] += rgb[gs, 2] * alpha * T
                #     contributor += 1
                #     T = test_T
                # contributor_gs_num[pixel_idx] = contributor
        rgb_for_tile = torch.zeros(TILE_X * TILE_Y, filtered_gs.shape[0])
        for gs in filtered_gs:
            T = torch.zeros((TILE_X * TILE_Y), dtype=torch.float32, device=points_xy_image.device)
            d = [points_xy_image[gs, 0] - pixel_for_tile[:, 1], points_xy_image[gs, 1] - pixel_for_tile[:, 2]]
            power = -0.5 * (conic_opacity[gs, 0] * d[:, 0] * d[:, 0] + conic_opacity[gs, 2] * d[:, 1] * d[:, 1]) - conic_opacity[gs, 1] * d[:, 0] * d[:, 1]
            power = torch.where(power < 0.001, torch.tensor(-float('inf')), power)
            alpha = conic_opacity[gs, 3] * torch.exp(power)
            T = T * (1 - alpha)
            T = torch.where(T < 0.0001, torch.tensor(0.0), T)
            rgb_for_tile += rgb[gs, :] * alpha[:, None] * T[:, None]
            
            
    rgb_for_pixel = rgb_for_pixel.cpu()

    # 将张量值从[0, 1]范围扩展到[0, 255]范围并转换为uint8类型
    rgb_for_pixel = (rgb_for_pixel * 255).byte()

    # 转换为PIL图像
    transform = transforms_T.ToPILImage()
    image = transform(rgb_for_pixel.permute(2, 0, 1))  # 需要将张量的通道顺序从(W, H, 3)转换为(3, W, H)

    # 保存图像
    image.save('output_image.png')
    end_time = time.time()
    print(f"渲染图片保存成功，用时为{end_time - start_time}")
        
    return

def IfTileInGS(idx, tile_x, rect_min, rect_max):
    idx_x = idx % tile_x
    idx_y = idx // tile_x
    X_in_gs = (idx_x >= rect_min[:, 0]) & (idx_x < rect_max[:, 0])
    Y_in_gs = (idx_y >= rect_min[:, 1]) & (idx_y < rect_max[:, 1])
    tile_in_gs = X_in_gs & Y_in_gs
    return tile_in_gs

def render_per_pixel(points_xy_image, rgb, conic_opacity, depths, background, tiles_x, tiles_y, W, H, rect_min, rect_max):
    start_time = time.time()
    
    output_color = torch.zeros((W, H, 3), dtype=torch.float, device='cuda')
    final_T = torch.zeros((W, H, 1), device=points_xy_image.device, dtype=torch.float) # 每一像素点的透明度
    contribute_num_for_pixel = torch.zeros((W, H, 1), dtype=torch.float, device='cuda')

    # 渲染进度条
    # progress_bar = tqdm(range(tiles_x * tiles_y), desc="Rendering progress")
    
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
        # if_idx_in_gs = ((i >= rect_min[:, 0]) & (i < rect_max[:, 0]) & (j >= rect_min[:, 1]) & (j < rect_max[:, 1]))
        depths_filtered = depths[if_idx_in_gs]
        # sorted_idx = torch.argsort(depths_filtered)
        filtered_gs_sorted = points_xy_image[if_idx_in_gs]
        # depth_tile = depths[if_idx_in_gs]
        
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

        alpha_cum = 1 - alpha
        T = torch.cumprod(alpha_cum, dim=2)
        T = T / (1 - alpha)
        
        final_T[pixel_min_x : pixel_max_x, pixel_min_y : pixel_max_y] = T[..., -1, 0:1] # 更新当前tile的final_T，final_T=π(i)(1-alpha)
        color = torch.sum(alpha * T * rgb_c, dim=2)
        output_color[pixel_min_x : pixel_max_x, pixel_min_y : pixel_max_y] = color
        contribute_num_for_pixel[pixel_min_x : pixel_max_x, pixel_min_y : pixel_max_y] = torch.sum((alpha != 0).int(), dim = 2)
    #         with torch.no_grad():
    #             progress_bar.update(1)
        
    # progress_bar.close()
    contribute_num_for_pixel_1D = contribute_num_for_pixel.squeeze(-1).flatten()
    _, ax = plt.subplots()
    ax.plot(contribute_num_for_pixel_1D.cpu(), marker='o', linestyle='-', color='b', label='Line 1')
    plt.title('GS contributed per pixel')
    ax.set_xlabel("Pix_Index")
    ax.set_ylabel("contribute_GS_num")
    plt.savefig("contribute_GS_num_per_PIX.png")
    
    output_color += final_T * background

    # 显示渲染得到的图片
    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
    output_color = output_color.permute(1, 0, 2)
    render_image = output_color.detach().cpu().numpy()

    # 确保 RGB 值在 [0, 1] 的范围内，并转换为 [0, 255] 的范围内的整数
    render_image = (np.clip(render_image, 0, 1) * 255).astype(np.uint8)

    # 将 NumPy 数组转换为图像
    render_image = Image.fromarray(render_image)

    # 显示图像
    render_image.show()

    # 将图像保存到文件
    # render_image.save("render_image.png")
    end_time = time.time()
    print(f"渲染图片保存成功，用时为{end_time - start_time}")
    
    return output_color
            
            
    
    
    
    
    
        
    
    