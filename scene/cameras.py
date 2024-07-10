import numpy as np
from torch import nn
import torch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans = np.array([0.0, 0.0, 0.0]), scale = 1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()
        
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.uid = uid
        
        try:
           self.data_device =  torch.device(data_device)
        except Exception as e:
        # 捕获所有可能的异常，并将异常信息赋值给变量 e
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
            
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)  # 使用 clamp 方法将 image 张量的值限制在 [0.0, 1.0] 范围内。
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=data_device)

            self.zfar = 100.0
            self.znear = 0.01
            
            self.trans = trans
            self.scale = scale
            
            self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        