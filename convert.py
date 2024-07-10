from argparse import ArgumentParser
import os
import logging
import shutil


parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
    # 当 exist_ok=True 时，如果目标目录已经存在，函数不会引发异常
    
    ## 特征提取
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    
    exit_code = os.system(feat_extracton_cmd) 
    # os.system()：用于在子shell中执行系统命令。它将构建好的命令字符串作为参数传递，并在命令完成后返回退出代码。
    # exit_code：存储命令的退出代码。如果命令成功执行，通常返回 0。非零值表示命令执行中发生了错误。
    
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
        
        
    ## 特征匹配 
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)
        
        
    ## 映射
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
        

## 图像去畸变
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)
    
files = os.listdir(args.source_path + "/sparse") 
# 获取 sparse 目录下的所有文件列表。
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)

for file in files:
    if file == '0':
        continue
    
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse" , "0", file)
    shutil.move(source_file, destination_file)
    
if args.resize:
    os.makedirs(args.source_path, "/image_2", exist_ok= True)
    os.makedirs(args.source_path, "/image_4", exist_ok= True)
    os.makedirs(args.source_path, "/image_8", exist_ok= True)
    
    files = os.listdir(args.source_path + "/images")
    
    for file in files:
        source_file = os.path.join(args.source_path, "/images", file)
        
        # 缩放为一半
        destination_file = os.path.join(args.source_path, "/images_2",file)
        shutil.copy2(source_file,destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        
        # 缩放为1/4
        destination_file = os.path.join(args.source_path, "/images_4",file)
        shutil.copy2(source_file,destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        
        # 缩放为1/8
        destination_file = os.path.join(args.source_path, "/images_8",file)
        shutil.copy2(source_file,destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
    
print("Done.")