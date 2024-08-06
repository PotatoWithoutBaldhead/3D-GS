import os
from scene.colmap_loader import read_extrinsics_binary, read_insrinsics_binary, read_points3D_binary, read_points3D_text
from scene.colmap_loader import read_extrinsics_text, read_insrinsics_text
from scene.colmap_loader import qvec2rotmat
from PIL import Image
import numpy as np
import sys
from utils.graphics_utils import focal2fov, getWorld2View2
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
import json
from pathlib import Path
from utils.sh_utils import SH2RGB



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud : BasicPointCloud
    train_cameras : list
    test_cameras : list
    nerf_normalization: dict
    ply_path : str

def storePly(ply_path, xyzs, rgbs):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
              ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
     
    normals = np.zeros_like(xyzs)
     
    elements = np.empty(xyzs.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzs, normals, rgbs), axis=1)
    elements[:] = list(map(tuple,attributes))
    
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)
    
def fetchPly(ply_path):
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points = positions, colors = colors, normals = normals)
        
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    # 从相机的外部和内部参数中读取相机信息，并将这些信息存储在 CameraInfo 对象中，
    # 最后返回一个包含所有相机信息的列表。
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()
        
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)  
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        
        ## 获取文件路径 image_path 中的文件名部分（不含扩展名）。
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,image=image,
                              image_path=image_path,image_name=image_name,width=width,height=height)
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
    # 计算相机中心的平均位置（中心）和相机中心到该平均位置的最大距离（对角线长度）
        cam_centers = np.hstack(cam_centers)  # 用于将数组沿水平方向堆叠。
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis = 0)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    
    cam_centers =[]
    
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])  # 为了保持维度一致，进行切片3:4
        
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    
    translate = -center
    
    return {"translate": translate, "radius" : radius}
        
def readColmapSceneInfo(path, images, eval, llffhold = 8):
    
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_insrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_insrinsics_binary(cameras_insrinsic_file)
    # try:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    #     cameras_insrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    #     cam_intrinsics = read_insrinsics_binary(cameras_insrinsic_file)
    # except:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    #     cameras_insrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_insrinsics_text(cameras_insrinsic_file)
        
    read_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics = cam_extrinsics, cam_intrinsics = cam_intrinsics, images_folder = os.path.join(path, read_dir))

    # 根据图像名称对相机信息进行排序
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_info = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_info = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        
        xyz, rgb, _ = read_points3D_binary(bin_path)
        # try:
        #     xyz, rgb, _ = read_points3D_binary(bin_path)
        # except:
        #     xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)  # 将点的坐标、法向量、颜色、写入ply文件
        
    pcd = fetchPly(ply_path)
        
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=cam_infos,
        test_cameras=test_cam_info,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    
    return scene_info

def readCamerasFromTransforms(path ,transformsfile, white_background, extension=".png"):
    cam_infos = []
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]  # 读取相机的水平视场角
        
        frames = contents["frames"]  # 获取所有帧的信息
        
        for idx,frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3,3]
            
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            
            im_data = np.array(image.convert("RGBA"))
            
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            
            ## 计算垂直视场角
            fovy = focal2fov(focal2fov(fovx, image.size[0]))
            FovY = fovy
            FovX = fovx
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos
              
def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # 用于读取 NeRF（Neural Radiance Fields）合成数据集的信息，包括相机参数和点云数据.
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # 没有colmap文件，随机初始化点云
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts,3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        storePly(ply_path, xyz, SH2RGB(shs) * 255.0)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info    
    
sceneLoadTypeCallbacks = {
    "Colmap" : readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}