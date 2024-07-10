import torch
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from plyfile import PlyData
import numpy as np
from torch import nn
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2



class GaussianModel:
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actral_covariance = L @ L.transpose(1,2)
            symm = strip_symmetric(actral_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        
        self.covariance_activation = build_covariance_from_scaling_rotation
        
        self.opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
            
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)    
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)    
    
    @property
    # @property 是 Python 中的一个内置装饰器，它使得类方法可以像属性一样被访问。
    # 使用 @property 可以让我们通过点操作符访问方法，而不需要加上括号。这对于封装和控制访问器（getter）方法特别有用，使代码更简洁、更直观。
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        
    def load_ply(self, path):
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), 
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device='cuda').requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1,2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1,2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device='cuda').requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device='cuda').requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device='cuda').requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree
              
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale  # 设置空间学习率缩放参数
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # 将点云数据转换为 torch tensor，并复制到 GPU 上
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors))).float().cuda()  # 将颜色数据转换为 SH (球谐函数) 表示，并复制到 GPU 上
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color  # 将颜色信息放到特征的第一个通道
        features[:, :3, 1:] = 0.0  # 其余通道初始化为0
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        ## 计算每个点到其他点的最小距离，并确保距离值不小于一个很小的数值
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        
        ## 根据距离计算尺度，并将尺度重复3次
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        
        ## 初始化旋转矩阵，默认没有旋转 (四元数表示法)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1
        
        ## 初始化不透明度，并将其值通过逆 sigmoid 函数转换
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1,2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True)) 
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params' : [self._xyz], 'lr' : training_args.position_lr_init * self.spatial_lr_scale, 'name' : 'xyz'},
            {'params' : [self._features_dc], 'lr' : training_args.feature_lr, 'name' : 'f_dc'},
            {'params' : [self._features_rest], 'lr' : training_args.feature_lr, 'name' : 'f_rest'},
            {'params' : [self._opacity], 'lr' : training_args.opacity_lr, 'name' : 'opacity'},
            {'params' : [self._scaling], 'lr' : training_args.scaling_lr, 'name' : 'scaling'},
            {'params' : [self._rotation], 'lr' : training_args.rotation_lr, 'name' : 'rotation'},
        ]
        
        self.optimizer = torch.optim.Adam(l, lr = 0.0, eps = 1e-15)
        self.xyz_schedular_args = get_expon_lr_func(lr_init = training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final = training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult = training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict) 
        
    def update_learning_rate(self, iter):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_schedular_args(iter)
                param_group['lr'] = lr
                return lr
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    
    