from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.system_utils import searchForMaxIteration
import os
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import camera_to_JSON
import json
import random
from utils.camera_utils import cameraList_from_camInfos


class Scene:
    
    gaussians : GaussianModel
    
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration = None, shuffle = True, resolution_scales = [1.0]):
        
        self.model_path = args.model_path
        self.load_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.load_iter = searchForMaxIteration(os.path.join(self.model_path, 'point_cloud'))
            else:
                self.load_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.load_iter))
            
        self.train_cameras = {}
        self.test_cameras = {}
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
            
        if not self.load_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)  # 将json_cams序列化为JSON格式，并写入文件。
            
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]
            
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        # if self.load_iter:
        #     self.gaussians.load_ply(os.path.join(self.model_path,"point_cloud",
        #                                                 "iteration_" + str(self.loaded_iter),
        #                                                 "point_cloud.ply"))
        
        if self.load_iter:
            self.gaussians.load_ply(os.path.join(os.getcwd(), "data", "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                
    def getTrainCameras(self, scale = 1.0):
        return self.train_cameras[scale]