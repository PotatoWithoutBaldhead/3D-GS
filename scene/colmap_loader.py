import struct
import collections
import numpy as np

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

# struct.unpack(format, buffer)
'''
format:一个字符串，定义了字节流的格式。它由格式字符组成，每个格式字符对应一种数据类型。常见的格式字符包括：
c: 字符(character),1 字节
b: 有符号字节(signed byte),1 字节
B: 无符号字节(unsigned byte),1 字节
?: 布尔值(boolean),1 字节
h: 有符号短整型(signed short),2 字节
H: 无符号短整型(unsigned short),2 字节
i: 有符号整型(signed int),4 字节
I: 无符号整型(unsigned int),4 字节
l: 有符号长整型(signed long),4 字节
L: 无符号长整型(unsigned long),4 字节
q: 有符号长长整型(signed long long),8 字节
Q: 无符号长长整型(unsigned long long),8 字节
f: 浮点数(float),4 字节
d: 双精度浮点数(double),8 字节
s: 字符串(string),1 字节每字符(s 后面的数字表示字符数，例如 10s 表示 10 个字符)
p: Pascal 字符串,1 字节每字符（字符串以第一个字节表示长度）
P: 指针(pointer),与平台相关(通常是 4 或 8 字节)
'''

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def qvec2rotmat(qvec):
    return np.array([
    [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
        2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
    [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
        1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
        2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
    [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
        2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
        1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy -Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy -Rxx -Rzz, 0, 0],
        [Rzx + Rxz, Rzx + Ryz, Rzz - Rxx -Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ])

    ## 使用 np.linalg.eigh 函数计算对称矩阵 K 的特征值（eigvals）和特征向量（eigvecs）。
    eigvals, eigvecs = np.linalg.eigh(K)
    
    ## eigvecs[[3, 0, 1, 2], np.argmax(eigvals)] 提取对应于最大特征值的特征向量
    ## 并重新排序以匹配四元数的标准格式 [w, x, y, z]。
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    
    ## 规范化四元数
    if qvec[0] < 0:
        qvec *= -1
    return qvec
    
    
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
def read_extrinsics_binary(path_to_model_file):
    """
    读取二进制文件中的图像数据。
    """    
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # 如果当前字符不是ASCII 0（字符串结束符）
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes= 24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            # 这里的数据表示一组二维点的坐标（x, y）和对应的3D点ID。这些数据的结构是交错的，即每组三个值分别是x坐标、y坐标和3D点ID。
            
            # 我们需要将这些数据重组为两个数组：一个是二维点的坐标数组 xys，另一个是3D点ID的数组 point3D_ids。
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(float, x_y_id_s[2::3])))
            
            images[image_id] = Image(
                id = image_id, qvec = qvec, tvec = tvec,
                camera_id = camera_id, name = image_name,
                xys = xys, point3D_ids = point3D_ids)
            
    return images
                   
def read_insrinsics_binary(path_to_model_file):
    """
    读取二进制文件中的内参数据。
    """ 
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            
            cameras[camera_id] = Camera(
                id = camera_id,
                model = model_name,
                width = width,
                height = height,
                params = np.array(params)
            )
        
        # 确保读取的相机数量与预期数量一致
        assert len(cameras) == num_cameras
    return cameras

def read_extrinsics_text(path):
    images ={}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = elems[0]
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3]))],[tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                
                images[image_id] = Image(
                    id = image_id, qvec = qvec, tvec = tvec,
                    camera_id = camera_id, name = image_name,
                    xys = xys, point3D_ids = point3D_ids
                )

def read_insrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = elems[2]
                height = elems[3]
                params = np.array(tuple(map(float, elems[4:])))
                
                cameras[camera_id] = Camera(
                    id = camera_id, model = model,
                    width = width, height = height,
                    params = params
                )
    return cameras

def read_points3D_binary(path_to_model_file):
    
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        
        for p_id in range(num_points):
            binary_point_line_poperties = read_next_bytes(
                fid, num_bytes= 43, format_char_sequence= "QdddBBBd")
            
            xyz = np.array(binary_point_line_poperties[1:4])
            rgb = np.array(binary_point_line_poperties[4:7])
            error = np.array(binary_point_line_poperties[7])
            
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
            
    return xyzs, rgbs, errors

def read_points3D_text(path_to_model_file):
    
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path_to_model_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1
                
    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    
    with open(path_to_model_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(float, elems[4:7])))
                error = np.array(float(elems[7]))
                
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1
    return xyzs, rgbs, errors
                
                
            
            
            