import sys
import random
import torch
import numpy as np
from datetime import datetime


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)  # 使用 PIL 的 resize 方法将图像调整为指定的分辨率
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0  # 转换为 NumPy 数组并归一化
    
    """
    检查 resized_image 的形状:
    ·如果图像是彩色的，则形状为 (高度, 宽度, 通道)。
    resized_image.permute(2, 0, 1) 将张量的维度从 (高度, 宽度, 通道) 调整为 (通道, 高度, 宽度)。
    ·如果图像是灰度图，则形状为 (高度, 宽度)。
    resized_image.unsqueeze(dim=-1) 在最后一维添加一个通道维度，使形状变为 (高度, 宽度, 1)。
    然后 permute(2, 0, 1) 将张量的维度调整为 (通道, 高度, 宽度)。
    """
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim = -1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    # 用来实现学习率的动态调整，从而提高训练的稳定性和效果。
    """
    lr_init (float): 初始学习率。
    lr_final (float): 最终学习率。
    lr_delay_steps (int, optional): 延迟学习率衰减的步数。默认值为0,表示没有延迟。
    lr_delay_mult (float, optional): 延迟期间的学习率乘数因子。默认值为1.0,表示没有额外延迟影响。
    max_steps (int, optional): 学习率衰减的最大步数。默认值为1000000。
    """
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        
        # 计算当前步数相对于最大步数的比例 t，并限制在0到1之间。
        t = np.clip(step / max_steps, 0, 1)
        
        # 对数线性插值法 log_lerp 计算当前步数对应的学习率，保证学习率在 lr_init 和 lr_final 之间平滑过渡
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    
    return helper

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    
    q = r / norm[:, None]  # 扩大 norm 的维度使其与 r 的形状兼容。
    
    R = torch.zeros((q.size(0), 3, 3), device= 'cuda')

    r = q[:,0]
    x = q[:,1]
    y = q[:,2]
    z = q[:,3]
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (y*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y) 
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype = torch.float, device= 'cuda')
    R = build_rotation(r)
    
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    
    L = R @ L
    return L

def strip_lowerdiag(L):
    # 它从一个下三角矩阵 L 中提取出对角线和次对角线的元素，并将它们存储在一个新的张量 uncertainty 中
    uncertainty = torch.zeros((L.shape[0], 6), dtype= torch.float, device='cuda')
    
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def inverse_sigmoid(x):
    return torch.log(x / (1-x))
    
def safe_state(silent):  
    # 该函数修改了标准输出（stdout）的行为，并设置了随机种子以确保可重复性。
    
    old_f = sys.stdout  
    # 将当前的标准输出流保存到 old_f，以便稍后可以恢复
    
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)
    # 更改标准输出流
    
    ## 设置随机种子和设备配置
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    

    