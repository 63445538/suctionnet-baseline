python pc_net/inference.py --split test_novel --camera realsense --network_ver 'v0.2.7.4' --epoch_num 40 --gpu_id 0 --sample_time 20 --save_root 'pc_net/save'
python eval.py --split test_novel --network_ver v0.2.7.4.p_40_20 --camera realsense