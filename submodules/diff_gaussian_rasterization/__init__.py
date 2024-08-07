from typing import NamedTuple
import torch
from torch import nn
from submodules.diff_gaussian_rasterization.rasterize import RasterizeGaussians


# def rasterize_gaussians(
#     means3D,
#     sh,
#     colors_precomp,
#     opacities,
#     scales,
#     rotations,
#     cov3Ds_precomp,
#     raster_settings
# ):
#     rasterizer = RasterizeGaussians(raster_settings)
#     return rasterizer(
#         means3D,
#         sh,
#         colors_precomp,
#         opacities,
#         scales,
#         rotations,
#         cov3Ds_precomp
#     )


class GaussianRasterizationSettings(NamedTuple):
    image_height : int
    image_width : int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    
class GaussianRasterizer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            from . import _C
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix
            )
        return visible
    
    def forward(self, means3D, opacities, shs = None, color_precomp = None,
                scales = None, rotations = None, cov3D_precomp = None, raster_settings = None):
        self.raster_settings = raster_settings
        
        if (shs is None and color_precomp is None) or (shs is not None and color_precomp is not None):
            raise Exception('Please provide excatly one either SHs or precomputed colors!')
    
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide excatly one either scales/rotations pair or precomputed covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if color_precomp is None:
            color_precomp = torch.Tensor([])
            
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
            
        rasterizer = RasterizeGaussians(raster_settings)

        ## 开始渲染
        color, radii, visibility_filter, p_view = rasterizer(
            means3D,
            shs,
            color_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp
        )
        p_view.retain_grad()
        self._p_view = p_view
        return color, radii, visibility_filter, p_view
    