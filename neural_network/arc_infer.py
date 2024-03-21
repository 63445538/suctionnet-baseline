import os
import torch
import time
import numpy as np
import open3d as o3d
import torch.nn as nn
import scipy.io as scio
import argparse
import DeepLabV3Plus.network as network
import ConvNet
import torch.nn.functional as F
from PIL import Image
import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='deeplabv3plus_resnet101', help='Model file name [default: votenet]')
parser.add_argument("--num_classes", type=int, default=1)
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
parser.add_argument('--log_dir', default='log_inf', help='Dump dir to save model checkpoint [default: log]')
# parser.add_argument('--split', default='test_seen', help='dataset split [default: test_seen]')
# parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/ARC', help='where dataset is')
parser.add_argument('--save_dir', default='/media/gpuadmin/rcao/result/arc/suctionnet/infer_result', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--checkpoint_path', default='/media/gpuadmin/rcao/result/arc/suctionnet/log/v0.1/checkpoint_90', help='Model checkpoint path [default: None]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--save_visu', action='store_true', default=True, help='whether to save visualizations.')
cfgs = parser.parse_args()

LOG_DIR = cfgs.log_dir
CHECKPOINT_PATH = cfgs.checkpoint_path
SAVE_PATH = cfgs.save_dir
# split = cfgs.split
# camera = cfgs.camera
vis_save_interval = 5
dataset_root = cfgs.dataset_root
scene_list = []

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'convnet_resnet101': ConvNet.convnet_resnet101,
        'deeplabv3plus_resnet101_depth': network.deeplabv3plus_resnet101_depth
    }
net = model_map[cfgs.model](num_classes=cfgs.num_classes, output_stride=cfgs.output_stride)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# net = nn.DataParallel(net)
net.to(device)
   

EPOCH_CNT = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    print('Loading model from:')
    print(CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH)
    
    net.load_state_dict(checkpoint['model_state_dict'])

    EPOCH_CNT = checkpoint['epoch']

def uniform_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    # center = kernel_size // 2
    kernel = kernel / kernel_size**2

    return kernel

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def grid_sample(pred_score_map, down_rate=20, topk=512):
    num_row = pred_score_map.shape[0] // down_rate
    num_col = pred_score_map.shape[1] // down_rate

    idx_list = []
    for i in range(num_row):
        for j in range(num_col):
            pred_score_grid = pred_score_map[i*down_rate:(i+1)*down_rate, j*down_rate:(j+1)*down_rate]
            
            max_idx = np.argmax(pred_score_grid)
            max_idx = np.array([max_idx // down_rate, max_idx % down_rate]).astype(np.int32)
            
            max_idx[0] += i*down_rate
            max_idx[1] += j*down_rate
            idx_list.append(max_idx[np.newaxis, ...])
    
    idx = np.concatenate(idx_list, axis=0)
    suction_scores = pred_score_map[idx[:, 0], idx[:, 1]]
    sort_idx = np.argsort(suction_scores)
    sort_idx = sort_idx[::-1]

    sort_idx_topk = sort_idx[:topk]

    suction_scores_topk = suction_scores[sort_idx_topk]
    idx0_topk = idx[:, 0][sort_idx_topk]
    idx1_topk = idx[:, 1][sort_idx_topk]

    return suction_scores_topk, idx0_topk, idx1_topk

def drawGaussian(img, pt, score, sigma=1):
    """Draw 2d gaussian on input image.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    pt: list or tuple
        A point: (x, y).
    sigma: int
        Sigma of gaussian distribution.
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.
    """
    # img = to_numpy(img)
    tmp_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * score
    g = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    tmp_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    img += tmp_img

def get_suction_from_heatmap(depth_img, heatmap, camera_info):
    suction_scores, idx0, idx1 = grid_sample(heatmap, down_rate=10, topk=1024)

    if len(depth_img.shape) == 3:
        depth_img = depth_img[..., 0]
    point_cloud = create_point_cloud_from_depth_image(depth_img, camera_info)
    
    suction_points = point_cloud[idx0, idx1, :]

    tic_sub = time.time()
    
    point_cloud = point_cloud.reshape(-1, 3)
    pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    pc_voxel_sampled = pc_o3d.voxel_down_sample(0.003)
    points_sampled = np.array(pc_voxel_sampled.points).astype(np.float32)
    points_sampled = np.concatenate([suction_points, points_sampled], axis=0)
    pc_voxel_sampled.points = o3d.utility.Vector3dVector(points_sampled)
    pc_voxel_sampled.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
    pc_voxel_sampled.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    pc_voxel_sampled.normalize_normals()
    pc_normals = np.array(pc_voxel_sampled.normals).astype(np.float32)
    suction_normals = pc_normals[:suction_points.shape[0], :]

    toc_sub = time.time()
    print('estimate normal time:', toc_sub - tic_sub)

    suction_arr = np.concatenate([suction_scores[..., np.newaxis], suction_normals, suction_points], axis=-1)

    return suction_arr, idx0, idx1

def inference_one_scene(scene_idx, scene_name):

    rgb_file = os.path.join(dataset_root, 'color-input', scene_name + '.png')
    depth_file = os.path.join(dataset_root, 'depth-input',  scene_name + '.png')
    meta_file = os.path.join(dataset_root, 'camera-intrinsics', scene_name + '.txt')
    
    intrinsics = np.loadtxt(meta_file)
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    
    width = 640
    height = 480
    s = 10000.0
    camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

    rgb = cv2.imread(rgb_file).astype(np.float32) / 255.0
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / s
    rgb, depth = torch.from_numpy(rgb), torch.from_numpy(depth)
    
    depth = torch.clamp(depth, 0, 1)
    if cfgs.model == 'convnet_resnet101':
        depth = depth.unsqueeze(-1).repeat([1, 1, 3])
        rgbd = torch.cat([rgb, depth], dim=-1).unsqueeze(0)
    elif 'depth' in cfgs.model:
        rgbd = depth.unsqueeze(-1).unsqueeze(0)
    else:
        rgbd = torch.cat([rgb, depth.unsqueeze(-1)], dim=-1).unsqueeze(0)
    
    rgbd = rgbd.permute(0, 3, 1, 2)
    rgbd = rgbd.to(device)

    net.eval()
    tic = time.time()
    with torch.no_grad():
        pred = net(rgbd)
    pred = pred.clamp(0, 1)
    toc = time.time()
    print('inference time:', toc - tic)

    # heatmap = (pred[0, 0] * pred[0, 1]).cpu().unsqueeze(0).unsqueeze(0)
    heatmap = (pred[0, 0]).cpu().unsqueeze(0).unsqueeze(0)
    
    k_size = 15
    kernel = uniform_kernel(k_size)
    kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
    heatmap = F.conv2d(heatmap, kernel, padding=(kernel.shape[2] // 2, kernel.shape[3] // 2)).squeeze().numpy()

    # suctions, idx0, idx1 = get_suction_from_heatmap(depth.numpy(), heatmap, camera_info)
    
    # save_dir = os.path.join(SAVE_PATH, split, 'scene_%04d'%scene_idx, camera, 'suction')
    # os.makedirs(save_dir, exist_ok=True)
    # suction_numpy_file = os.path.join(save_dir, '%04d.npz'%anno_idx)
    # print('Saving:', suction_numpy_file)
    # np.savez(suction_numpy_file, suctions)

    if scene_idx % vis_save_interval and cfgs.save_visu:
        rgb_img = rgbd[0].permute(1, 2, 0)[..., :3].cpu().numpy()
        rgb_img *= 255
        
        # predictions
        score = pred[0, 0].clamp(0, 1).cpu().numpy()
        # center = pred[0, 1].clamp(0, 1).cpu().numpy()

        score *= 255
        # center *= 255
        # mix = heatmap * 255

        score_img = cv2.applyColorMap(score.astype(np.uint8), cv2.COLORMAP_RAINBOW)
        score_img = score_img * 0.5 + rgb_img * 0.5
        score_img = score_img.astype(np.uint8)
        score_img = Image.fromarray(score_img)

        # center_img = cv2.applyColorMap(center.astype(np.uint8), cv2.COLORMAP_RAINBOW)
        # center_img = center_img * 0.5 + rgb_img * 0.5
        # center_img = center_img.astype(np.uint8)
        # center_img = Image.fromarray(center_img)

        # mix_img = cv2.applyColorMap(mix.astype(np.uint8), cv2.COLORMAP_RAINBOW)
        # mix_img = mix_img * 0.5 + rgb_img * 0.5
        # mix_img = mix_img.astype(np.uint8)
        # mix_img = Image.fromarray(mix_img)

        score_dir = os.path.join(SAVE_PATH, 'visu_90')
        os.makedirs(score_dir, exist_ok=True)
        score_file = os.path.join(score_dir, scene_name + '.png')
        print('saving:', score_file)
        score_img.save(score_file)

        # center_dir = os.path.join(SAVE_PATH, split, 'scene_%04d'%scene_idx, camera, 'visu')
        # os.makedirs(center_dir, exist_ok=True)
        # center_file = os.path.join(center_dir, '%04d_center.png'%anno_idx)
        # print('saving:', center_file)
        # center_img.save(center_file)

        # mix_dir = os.path.join(SAVE_PATH, split, 'scene_%04d'%scene_idx, camera, 'visu')
        # os.makedirs(mix_dir, exist_ok=True)
        # mix_file = os.path.join(mix_dir, '%04d_mix.png'%anno_idx)
        # print('saving:', mix_file)
        # mix_img.save(mix_file)

        # sampled suctions
        # sampled_img = np.zeros_like(heatmap)
        # for i in range(suctions.shape[0]):
        #     drawGaussian(sampled_img, [idx1[i], idx0[i]], suctions[i, 0], 3)
        
        # sampled_img *= 255
        # sampled_img = cv2.applyColorMap(sampled_img.astype(np.uint8), cv2.COLORMAP_RAINBOW)
        # sampled_img = sampled_img * 0.5 + rgb_img * 0.5
        # sampled_img = sampled_img.astype(np.uint8)
        # sampled_img = Image.fromarray(sampled_img)
        
        # sampled_dir = os.path.join(SAVE_PATH, split, 'scene_%04d'%scene_idx, camera, 'visu')
        # os.makedirs(sampled_dir, exist_ok=True)
        # sampled_file = os.path.join(sampled_dir, '%04d_sampled.png'%anno_idx)
        # print('saving:', sampled_file)
        # sampled_img.save(sampled_file)
    return heatmap


scene_list = []
with open(os.path.join(dataset_root, 'test-split.txt')) as f:
    for line in f:
        scene_list.append(line.strip())

results = []
for scene_idx, scene_name in enumerate(scene_list):
    score = inference_one_scene(scene_idx, scene_name)
    results.append(score)

# 创建一个(367, 1)的object数组来存储这些数组
matlab_style_results = np.empty((len(results), 1), dtype=object)
for i, arr in enumerate(results):
    matlab_style_results[i, 0] = arr
    
scio.savemat('suctionnet_results_90.mat', {'results': matlab_style_results})