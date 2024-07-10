from argparse import ArgumentParser, Namespace
import os
import sys


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser : ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key[0] == '_':
                shorthand = True
                key = key[1:]

            t = type(value)
            value = value if not fill_none else None
            
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0]), default = value, action = "store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0]), default = value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default = value, action = "store_true")
                else:
                    group.add_argument("--" + key, default = value, type = t)
            
    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or '_'+arg[0] in vars(self):
                setattr(group,arg[0],arg[1])  # setattr(object, name, value) 为对象设置属性，若该属性名不存在则创建
        return group
    
    
class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)  # 将 g.source_path 转换为绝对路径，并更新 g.source_path
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

    
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")
        
        
def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]  # 获取命令行参数，跳过脚本名称
    cfgfile_string = "Namespace()"  # 初始化一个字符串 cfgfile_string，表示一个空的 Namespace 对象。这个字符串将在没有找到配置文件时使用。
    args_cmdline = parser.parse_args(cmdlne_string)  # 使用解析器 parser 解析命令行参数，将结果存储在 args_cmdline 对象中。
    
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")  # 拼接文件路径
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    
    args_cfgfile = eval(cfgfile_string)  # 用 eval 函数将 cfgfile_string 转换为 Namespace 对象 args_cfgfile。如果没有找到配置文件，cfgfile_string 将是一个空的 Namespace。
    
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    # 遍历 args_cmdline 中的所有参数及其值。如果参数值不为 None,则将其添加到 merged_dict 中。
    # 如果 args_cmdline 中的某个参数值覆盖了 args_cfgfile 中的相同参数值，则以 args_cmdline 的值为准。        
    
    return Namespace(**merged_dict)
'''
get_combined_args 函数的作用是：
1.从命令行获取参数。
2.尝试从指定路径读取配置文件中的参数。
3.将命令行参数和配置文件参数合并，以命令行参数优先。
4.返回合并后的参数作为一个 Namespace 对象。
'''