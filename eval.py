from ast import dump
from suctionnetAPI import SuctionNetEval
import os
import numpy as np
import argparse
# network_ver = 'v0.2.1.s'

import random
def setup_seed(seed):
     np.random.seed(seed)
     random.seed(seed)
# 设置随机数种子
setup_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='dataset split [default: test_seen]')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument('--network_ver', default='geo_v0.2', help='where to save')
parser.add_argument('--dataset_root', default='/data/rcao/dataset/graspnet', help='where dataset is')
FLAGS = parser.parse_args()

print(FLAGS)

if __name__ == "__main__":
    dataset_root = FLAGS.dataset_root
    camera = FLAGS.camera
    network_ver = FLAGS.network_ver
    split = FLAGS.split
    suctionnet_eval = SuctionNetEval(root=dataset_root, camera=camera)

    result_path = os.path.join('pc_net', 'save', network_ver)
    # evaluate all the test splits
    # res is the raw evaluation results, ap_top50 and ap_top1 are average precision of top 50 and top 1 suctions
    # see our paper for details
    if split == 'test_seen':
        res, ap_top50, ap_top1 = suctionnet_eval.eval_seen(dump_folder=result_path, proc=30)
    elif split == 'test_similar':
        res, ap_top50, ap_top1 = suctionnet_eval.eval_similar(dump_folder=result_path, proc=30)
    else:
        res, ap_top50, ap_top1 = suctionnet_eval.eval_novel(dump_folder=result_path, proc=30)
        
    save_path = os.path.join('pc_net', 'save', network_ver, 'dump_file', camera)
    os.makedirs(save_path, exist_ok=True)
    dump_file_save_path = os.path.join(save_path, '{}.npy'.format(split))
    np.save(dump_file_save_path, res)