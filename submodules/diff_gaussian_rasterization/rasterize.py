import torch
from torch import nn
from submodules.diff_gaussian_rasterization.preform_gaussian import preprocess, render_per_pixel

TILE_X = 16
TILE_Y = 16

NUM_CHANNELS = 3

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)
    
def resizeFunctional(tensor : torch.Tensor):
    def resizetensor(N):
        tensor.resize_((N,))
        return tensor.contiguous().data_ptr()
    return resizetensor

class RasterizeGaussians(nn.Module):
    def __init__(self, raster_settings):
        super(RasterizeGaussians, self).__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp
    ):
        # Restructure arguments the way that the Python rasterizer expects them
        args = (
            self.raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            self.raster_settings.scale_modifier,
            cov3Ds_precomp,
            self.raster_settings.viewmatrix,
            self.raster_settings.projmatrix,
            self.raster_settings.tanfovx,
            self.raster_settings.tanfovy,
            self.raster_settings.image_height,
            self.raster_settings.image_width,
            sh,
            self.raster_settings.sh_degree,
            self.raster_settings.campos,
            self.raster_settings.prefiltered,
            self.raster_settings.debug
        )

        # Invoke Python rasterizer
        if self.raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_gaussians_python(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occurred in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_gaussians_python(*args)

        # Save tensors for potential backward pass
        self.save_for_later(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        
        return tuple(num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer)

    def save_for_later(self, *tensors):
        self.saved_tensors = tensors
                
def rasterize_gaussians_python(
    background : torch.Tensor,
    means3D : torch.Tensor,
    colors : torch.Tensor,
    opacity : torch.Tensor,
    scales : torch.Tensor,
    rotations : torch.Tensor,
    scale_modifier : float,
    cov3D_precomp : torch.Tensor,
    viewmatrix : torch.Tensor,
    projmatrix : torch.Tensor,
    tan_fovx : float,
    tan_fovy : float,
    image_height : int,
    image_width : int,
    sh : torch.Tensor,
    degree : int,
    campos : torch.Tensor,
    prefiltered : bool,
    debug : bool
):
    if means3D.ndimension() != 2 or means3D.size(1) != 3:
        raise ValueError("means3D must have dimensions (num_points, 3)")
    
    P = means3D.size(0)
    H = image_height
    W = image_width

    rendered = 0
    if P != 0:
        M = 0
        if sh.size(0) != 0:
            M = sh.size(1)
        
        rendered = forward(
           P, degree, M,  # degree为实际生效的球谐阶数，M为参数个数
           background,
           W,H,
           means3D,
           sh,
           colors,
           opacity,
           scales,
           scale_modifier,
           rotations,
           cov3D_precomp,
           viewmatrix,
           projmatrix,
           campos,
           tan_fovx,
           tan_fovy,
           prefiltered,
           debug
        )
    return rendered

def forward(
    P, D, M, background,
    width, height,
    means3D, shs,
    colors_precomp,
    opacities,
    scales,
    scale_modifier,
    rotations,
    cov3D_precomp,
    viewmatrix,
    projmatrix,
    cam_pos,
    tan_fovx, tan_fovy,
    prefiltered,
    debug
):
    focal_y = height / (2.0 * tan_fovy)
    focal_x = width / (2.0 * tan_fovx)
    
    if NUM_CHANNELS != 3 and colors_precomp is None:
        raise RuntimeError("For non-RGB, provide precomputed Gaussian colors!")

    # 光栅化之前对每个高斯进行初步处理（包括计算每个高斯所占据的tile）
    depths, points_xy_image, rgb, conic_opacity, rect_min, rect_max = preprocess(P, D, M,
    means3D, scales, scale_modifier, rotations, opacities, shs, cov3D_precomp, 
    colors_precomp, viewmatrix, projmatrix, cam_pos, width, height, tan_fovx, tan_fovy, focal_x, focal_y, prefiltered
)
    tile_x = (width + TILE_X - 1) // TILE_X
    tile_y = (height + TILE_Y - 1) // TILE_Y

    # 按照depth对高斯的索引进行排序
    depth_sorted, sorted_indices = torch.sort(depths)

    output_color = render_per_pixel(points_xy_image[sorted_indices],
                                    rgb[sorted_indices], conic_opacity[sorted_indices], 
                                    depth_sorted, background, tile_x, tile_y, width, height, 
                                    rect_min[sorted_indices], rect_max[sorted_indices])
    
    
    return
    