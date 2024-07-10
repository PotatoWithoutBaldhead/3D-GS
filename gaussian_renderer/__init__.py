from scene.gaussian_model import GaussianModel
import torch
import math
from submodules.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh


def render(viewpoint_cameras, pc : GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier = 1.0, override_color = None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad = True, device = "cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # 创建光栅化配置
    tanfovx = math.tan(viewpoint_cameras.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cameras.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height = int(viewpoint_cameras.image_height),
        image_width = int(viewpoint_cameras.image_width),
        tanfovx = tanfovx,
        tanfovy = tanfovy,
        bg = bg_color,
        scale_modifier = scaling_modifier,
        viewmatrix = viewpoint_cameras.world_view_transform,
        projmatrix = viewpoint_cameras.full_proj_transform,
        sh_degree = pc.active_sh_degree,
        campos = viewpoint_cameras.camera_center,
        prefiltered = False,
        debug = pipe.debug
    ) 
    
    rasterizer = GaussianRasterizer(raster_settings = raster_settings)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    # 若python中预先计算了协方差，则直接使用，否则将在光栅化中计算
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        
    # 若python中预先计算了颜色，则直接使用；否则，若需要用python计算，则执行下述功能，
    # 若不需要python计算，则在光栅化中计算 SH -> RGB
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            
            ## 将点云的 SH 特征从形状 (N, (max_sh_degree + 1)^2, 3) 转换为 (N, 3, (max_sh_degree + 1)^2)
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            
            dir_pp = pc.get_xyz - viewpoint_cameras.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim = 1, keepdim = True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        color_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    return {
        "render" : rendered_image,
        "viewspace_points" : screenspace_points,
        "visibility_filter" : radii > 0,
        "radii" : radii
    }
    