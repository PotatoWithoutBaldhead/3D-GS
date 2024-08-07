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
        # self.means2D = None

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
                color, radii, visibility_filter, p_view = self.rasterize_gaussians_python(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occurred in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            color, radii, visibility_filter, p_view = self.rasterize_gaussians_python(*args)
        p_view.retain_grad()
        return color, radii, visibility_filter, p_view
                
    def rasterize_gaussians_python(self, 
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
        
        if P != 0:
            focal_y = H / (2.0 * tan_fovy)
            focal_x = W / (2.0 * tan_fovx)
            
            if NUM_CHANNELS != 3 and colors is None:
                raise RuntimeError("For non-RGB, provide precomputed Gaussian colors!")
            
            # 光栅化之前对每个高斯进行初步处理（包括计算每个高斯所占据的tile）
            depths, points_xy_image, rgb, conic_opacity, rect_min, rect_max, radii, visibility_filter, p_view = preprocess(P, degree,
            means3D, scales, scale_modifier, rotations, opacity, sh, cov3D_precomp, 
            colors, viewmatrix, projmatrix, campos, W, H, tan_fovx, tan_fovy, focal_x, focal_y, prefiltered
        )
            tile_x = (W + TILE_X - 1) // TILE_X
            tile_y = (H + TILE_Y - 1) // TILE_Y

            # 按照depth对高斯的索引进行排序
            depth_sorted, sorted_indices = torch.sort(depths)

            output_color = render_per_pixel(points_xy_image[sorted_indices],
                                            rgb[sorted_indices], conic_opacity[sorted_indices], 
                                            depth_sorted, background, tile_x, tile_y, W, H, 
                                            rect_min[sorted_indices], rect_max[sorted_indices])
            # self.means2D = points_xy_image.retain_grad()
            p_view.retain_grad()
            return output_color, radii, visibility_filter, p_view
