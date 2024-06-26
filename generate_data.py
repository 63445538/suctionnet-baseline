import numpy as np
import open3d as o3d

import os
from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from pytorch3d.ops.knn import knn_points

import torch
from suctionnetAPI.utils.xmlhandler import xmlReader
from suctionnetAPI.utils.eval_utils import parse_posevector, eval_suction, create_table_points, voxel_sample_points, \
    transform_points, get_scene_name

import argparse
# from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
partial_num = 100000

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/data/rcao/dataset/graspnet')
parser.add_argument('--camera_type', default='realsense', help='Camera split [realsense/kinect]')


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def loadSealLabels(objIds=None):
    '''
    **Input:**
    - objIds: int or list of int of the object ids.
    **Output:**
    - a dict of seal labels of each object.
    '''
    # load object-level grasp labels of the given obj ids
    assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
    objIds = objIds if _isArrayLike(objIds) else [objIds]
    graspLabels = {}
    # for i in tqdm(objIds, desc='Loading seal labels...'):
    for i in objIds:
        file = np.load(os.path.join(dataset_root, 'seal_label', '{}_seal.npz'.format(str(i).zfill(3))))
        graspLabels[i] = (file['points'].astype(np.float32), file['normals'].astype(np.float32), file['scores'].astype(np.float32))
    return graspLabels


def loadWrenchLabels(sceneIds):
    '''
    **Input:**

    - sceneIds: int or list of int of the scene ids.
    **Output:**
    - dict of the wrench labels.
    '''
    assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
    sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
    wrenchLabels = {}
    # for sid in tqdm(sceneIds, desc='Loading wrench labels...'):
    for sid in sceneIds:
        labels = np.load(os.path.join(dataset_root, 'wrench_label', '%04d_wrench.npz' % sid))
        wrenchLabel = []
        for j in range(len(labels)):
            wrenchLabel.append(labels['arr_{}'.format(j)])
        wrenchLabels['scene_'+str(sid).zfill(4)] = wrenchLabel
    return wrenchLabels


def loadCollisionLabels(sceneIds):
    '''
    **Input:**

    - sceneIds: int or list of int of the scene ids.
    **Output:**
    - dict of the collision labels.
    '''
    assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
    sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
    collisionLabels = {}
    # for sid in tqdm(sceneIds, desc='Loading collision labels...'):
    for sid in sceneIds:
        labels = np.load(os.path.join(dataset_root, 'suction_collision_label', '%04d_collision.npz' % sid))
        collisionLabel = []
        for j in range(len(labels)):
            collisionLabel.append(labels['arr_{}'.format(j)])
        collisionLabels['scene_'+str(sid).zfill(4)] = collisionLabel
    return collisionLabels


def get_model_poses(camera, scene_id, ann_id):
    '''
    **Input:**

    - scene_id: int of the scen index.

    - ann_id: int of the annotation index.

    **Output:**

    - obj_list: list of int of object index.

    - pose_list: list of 4x4 matrices of object poses.

    - camera_pose: 4x4 matrix of the camera pose relative to the first frame.

    - align mat: 4x4 matrix of camera relative to the table.
    '''
    scene_dir = os.path.join(dataset_root, 'scenes')
    camera_poses_path = os.path.join(dataset_root, 'scenes', get_scene_name(scene_id), camera, 'camera_poses.npy')
    camera_poses = np.load(camera_poses_path)
    camera_pose = camera_poses[ann_id]
    align_mat_path = os.path.join(dataset_root, 'scenes', get_scene_name(scene_id), camera, 'cam0_wrt_table.npy')
    align_mat = np.load(align_mat_path)
    scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), camera, 'annotations', '%04d.xml'% (ann_id,)))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        pose_list.append(mat)
    return obj_list, pose_list, camera_pose, align_mat


def point_matching(scene_points, grasp_points, seal_scores, wrench_scores):
    grasp_points = grasp_points.contiguous().unsqueeze(0)
    scene_points_num = scene_points.shape[0]
    scene_seal_scores = np.zeros((scene_points_num, 1))
    scene_wrench_scores = np.zeros((scene_points_num, 1))
    part_num = int(scene_points_num / partial_num)
    for i in range(1, part_num + 2):   # lack of cuda memory
        if i == part_num + 1:
            cloud_masked_partial = scene_points[partial_num * part_num:]
            if len(cloud_masked_partial) == 0:
                break
        else:
            cloud_masked_partial = scene_points[partial_num * (i - 1):(i * partial_num)]
        cloud_masked_partial = torch.from_numpy(cloud_masked_partial).to(device).float()
        cloud_masked_partial = cloud_masked_partial.contiguous().unsqueeze(0)
        _, nn_inds, _ = knn_points(cloud_masked_partial, grasp_points, K=1)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)
        scene_seal_scores[partial_num * (i - 1):(i * partial_num)] = torch.index_select(seal_scores, 0, nn_inds).cpu().numpy()
        scene_wrench_scores[partial_num * (i - 1):(i * partial_num)] = torch.index_select(wrench_scores, 0, nn_inds).cpu().numpy()
    return scene_seal_scores, scene_wrench_scores


# if __name__ == '__main__':
cfgs = parser.parse_args()
dataset_root = cfgs.dataset_root   # set dataset root
camera_type = cfgs.camera_type   # kinect / realsense
save_path_root = os.path.join(dataset_root, 'suction')

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
for scene_id in range(160, 190):
    for ann_id in range(256):
        # get scene point cloud
        print('generating scene: {} ann: {}'.format(scene_id, ann_id))
        rgb = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                               camera_type, 'rgb', str(ann_id).zfill(4) + '.png')))
        depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                 camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
        seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                               camera_type, 'label', str(ann_id).zfill(4) + '.png')))

        meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                         camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        # # remove outlier and get objectness label
        # depth_mask = (depth > 0)
        # camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
        #                                     camera_type, 'camera_poses.npy'))
        # camera_pose = camera_poses[ann_id]
        # align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
        #                                  camera_type, 'cam0_wrt_table.npy'))
        #
        # workspace_mask = get_workspace_mask(cloud, seg, trans=np.dot(align_mat, camera_pose), organized=True, outlier=0.02)
        # mask = (depth_mask & workspace_mask)
        #
        # # get scene object and grasp info
        # scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
        #                                       camera_type, 'annotations', '%04d.xml' % ann_id))
        # pose_vectors = scene_reader.getposevectorlist()
        # obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)

        obj_list, pose_list, camera_pose, align_mat = get_model_poses(camera_type, scene_id, ann_id)
        workspace_mask = get_workspace_mask(cloud, seg=seg, trans=np.dot(align_mat, camera_pose), organized=True, outlier=0.02)

        depth_mask = (depth > 0)
        mask = (depth_mask & workspace_mask)
        print(obj_list)

        rgb_masked = rgb[mask]
        cloud_masked = cloud[mask]
        objectness_label = seg[mask]

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(cloud_masked.reshape((-1, 3)))
        scene.colors = o3d.utility.Vector3dVector(rgb_masked/255.0)
        downsampled_scene = scene.voxel_down_sample(voxel_size=0.003)

        # grasp_labels = {}
        # for i in obj_list:
        #     grasp_file = np.load(os.path.join(dataset_root, 'grasp_label_simplified', '{}_labels.npz'.format(str(i).zfill(3))))
        #     grasp_labels[i] = (grasp_file['points'].astype(np.float32), grasp_file['width'].astype(np.float32),
        #                        grasp_file['scores'].astype(np.float32))
        #     seal_file = np.load(os.path.join(dataset_root, 'seal_label', '{}_seal.npz'.format(str(i).zfill(3))))
        #
        #     seal_labels[i] =
        #
        seal_labels = loadSealLabels(obj_list)
        wrench_labels = loadWrenchLabels(scene_id)
        wrench_dump = wrench_labels['scene_{:04}'.format(scene_id)]

        # collision_labels = loadCollisionLabels(scene_id)
        # collision_dump = collision_labels['scene_{:04}'.format(scene_id)]

        # labels = np.load(os.path.join(dataset_root, 'collision_label', 'scene_' + str(scene_id).zfill(4), 'collision_labels.npz'))
        # collision_dump = []
        # for j in range(len(labels)):
        #     collision_dump.append(labels['arr_{}'.format(j)])
        grasp_points = []
        grasp_points_seal = []
        grasp_points_wrench = []
        inst_vis_list = []
        for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
            sampled_points, sampled_normals, seal_scores = seal_labels[obj_idx]
            wrench_scores = wrench_dump[i]

            target_points = transform_points(sampled_points, trans)
            grasp_points.append(target_points)
            grasp_points_seal.append(seal_scores.reshape(-1, 1))
            grasp_points_wrench.append(wrench_scores.reshape(-1, 1))
            # grasp_points_graspness.append(graspness.reshape(num_points, 1))

            inst_vis = o3d.geometry.PointCloud()
            inst_vis.points = o3d.utility.Vector3dVector(target_points.reshape((-1, 3)))
            inst_vis_list.append(inst_vis)

        grasp_points = np.vstack(grasp_points)
        grasp_points_seal = np.vstack(grasp_points_seal)
        grasp_points_wrench = np.vstack(grasp_points_wrench)

        # grasp_points_graspness = np.vstack(grasp_points_graspness)
        #
        grasp_points = torch.from_numpy(grasp_points).to(device)
        grasp_points_seal = torch.from_numpy(grasp_points_seal).to(device)
        grasp_points_wrench = torch.from_numpy(grasp_points_wrench).to(device)

        cloud_masked_seal_scores, cloud_masked_wrench_scores = point_matching(cloud_masked, grasp_points,
                                                                              grasp_points_seal, grasp_points_wrench)

        # max_seal_scores = np.max(cloud_masked_seal_scores)
        # min_seal_scores = np.min(cloud_masked_seal_scores)
        # cloud_masked_seal_scores = (cloud_masked_seal_scores - min_seal_scores) / (max_seal_scores - min_seal_scores)
        cloud_masked_seal_scores[~objectness_label.astype(bool), :] = 0

        # max_wrench_scores = np.max(cloud_masked_wrench_scores)
        # min_wrench_scores = np.min(cloud_masked_wrench_scores)
        # cloud_masked_wrench_scores = (cloud_masked_wrench_scores - min_wrench_scores) / (max_wrench_scores - min_wrench_scores)
        cloud_masked_wrench_scores[~objectness_label.astype(bool), :] = 0

        # cmap = plt.get_cmap('viridis')
        # cmap_rgb = cmap(cloud_masked_seal_scores[:, 0])[:, :3]
        # scene.colors = o3d.utility.Vector3dVector(cmap_rgb)
        # o3d.visualization.draw_geometries([origin_frame, scene] + inst_vis_list)

        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez_compressed(os.path.join(save_path, str(ann_id).zfill(4) + '.npz'),
                            seal_score=cloud_masked_seal_scores, wrench_score=cloud_masked_wrench_scores)
