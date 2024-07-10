from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import sys
from utils.general_utils import safe_state
from gaussian_renderer import render, network_gui
import torch
import os
import uuid
from scene import Scene
from scene.gaussian_model import GaussianModel
from tqdm import tqdm
from random import randint

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    gaussians = GaussianModel(3)
    scene = Scene(dataset, gaussians, load_iteration = True, shuffle=False)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack[0]
    bg = torch.rand((3), device="cuda") if opt.random_background else background
        
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

def training1(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        ## torch.load()用来加载torch.save() 保存的模型文件
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    ## 精确记录GPU上模型的推理时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        iter_start.record()
        
        gaussians.update_learning_rate(iteration)
        
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        ## 随机选择一个相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        ## render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)



def prepare_output_and_logger(args):
    
    ## 通过这种方式，函数可以确保输出路径的唯一性和可辨识性。
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder = {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok= True)
    
    ## 将 args 对象的内容写入 cfg_args 文件中。Namespace(**vars(args)) 
    ## 创建一个新的 Namespace 对象，str() 将其转换为字符串格式，最后写入文件。
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress.")
    
    return tb_writer
    
if __name__ == "__main__":
    # Set up command line argument parser 设置命令行参数解释器
    parser = ArgumentParser(description = "Train script parameters")  # 设置命令行参数解析器 parser
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1") 
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])  # 解析命令行参数并将其存储在args对象中
    args.save_iterations.append(args.iterations)  # 将迭代次数添加到保存迭代次数列表中
    
    print("Optimizing " + args.model_path)  # 输出要优化的模型路径
    
    # Initialize system state (RNG)
    safe_state(args.quiet)  # 初始化系统状态，如果设置了quiet参数，减少输出信息。这通常用于设置随机数生成器的状态以确保结果的可重复性
    
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    
    print("\nTraining complete.")