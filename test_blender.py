
import blenderproc as bproc
import numpy as np
import trimesh
import pickle

def get_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    value = data['00020346']
    print(value)

view_file = "/home/lkh/siga/CADIMG/datasets/render_normal/centers.pkl"
get_pkl(view_file)
raise Exception("stop")


VIEW_CORNERS = [
    [ 3.0, -3.0,  3.0],
    [ 3.0, -3.0, -3.0],
    [ 3.0,  3.0, -3.0],
    [ 3.0,  3.0,  3.0],
    [-3.0, -3.0,  3.0],
    [-3.0, -3.0, -3.0],
    [-3.0,  3.0, -3.0],
    [-3.0,  3.0,  3.0],
]

VIEW_TRANSFORM = [
    [   np.pi/4,  np.pi/4, 0],
    [ 3*np.pi/4, -np.pi/4, 0],
    [-3*np.pi/4, -np.pi/4, 0],
    [-  np.pi/4,  np.pi/4, 0],
    [   np.pi/4, -np.pi/4, 0],
    [ 3*np.pi/4,  np.pi/4, 0],
    [-3*np.pi/4,  np.pi/4, 0],
    [-  np.pi/4, -np.pi/4, 0],
]


VIEW = {
    'front': [0, 1, 2, 3],
    'back': [4, 5, 6, 7],
    'right': [2, 3, 6, 7],
    'left': [0, 1, 4, 5],
    'up': [0, 3, 4, 7],
    'down': [1, 2, 5, 6]
}

def look_at_rotation(camera_pos, target, up=np.array([0, 0, 1])):
    """
    计算一个从 camera_pos 看向 target 的旋转矩阵（3x3）
    """
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    true_up = np.cross(forward, right)
    true_up = true_up / np.linalg.norm(true_up)

    # Blender camera looks along -Z, so invert forward
    rotation = np.stack([-right, true_up, -forward], axis=1)
    return rotation

# 设置相机位置和朝向目标点





bproc.init()



# Create a simple object:
# file_path = "/home/lkh/siga/CADIMG/test/output_mesh.ply"
file_path = "/home/lkh/siga/CADIMG/test/ply/result/00020346.ply"
objs = bproc.loader.load_obj(file_path)
mesh = trimesh.load(file_path)
aabb = mesh.bounding_box
min_corner = aabb.bounds[0]  # [x_min, y_min, z_min]
max_corner = aabb.bounds[1]  # [x_max, y_max, z_max]
aabb = mesh.bounding_box
center = (min_corner + max_corner) /2 
obj=objs[0]
# set location
# obj.set_location([-center[0],-center[1],-center[2]])
obj.set_location([0,0,0])
# obj.set_rotation_euler([np.pi/2, np.pi/2, np.pi/2])
camera_pos = np.array([-3.0, 3.0, -3.0])
target_pos = np.array([0.0, 0.0, 0.0])

# 计算旋转矩阵
rotation = look_at_rotation(camera_pos, center)

# Set the camera to be in front of the object
'''
相机默认 视角向下看[0,0,-1]
世界坐标系是
0   -1    0
1    0    0
0    0    1
旋转矩阵[a,b,c]是围绕上面的坐标系逆时针旋转的
'''

view = 'left'
camera_list = VIEW[view]
focus_point = bproc.object.create_empty("Camera Focus Point")
focus_point.set_location(center)
for i in camera_list:
    
    # 旋转矩阵
    camera_pos =  np.array(VIEW_CORNERS[i]) * 5/6
    rotation = look_at_rotation(camera_pos, center)
    cam_pose = bproc.math.build_transformation_mat(camera_pos, rotation)
    bproc.camera.add_depth_of_field(focus_point, fstop_value=1)
    bproc.camera.add_camera_pose(cam_pose)


# cam_pose = bproc.math.build_transformation_mat([-4, -4, -4], [-np.pi/4+np.pi, np.pi/4, 0])  # [绕y，绕x，绕z] 逆时针
# cam_pose1 = bproc.math.build_transformation_mat(camera_pos, rotation)
# cam_pose2 = bproc.math.build_transformation_mat([0, -3, 0], [np.pi/2, 0, 0])

# bproc.camera.set_resolution(512, 512)
# bproc.camera.add_camera_pose(cam_pose1)
# bproc.camera.add_camera_pose(cam_pose2)
bproc.renderer.enable_normals_output()
# Render the scene
data = bproc.renderer.render()

# Write the rendering into a hdf5 file
bproc.writer.write_hdf5("output/", data)

