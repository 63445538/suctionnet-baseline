# python eval.py --split test_seen --network_ver v0.2.7.4.p_40_4 --camera realsense
# python eval.py --split test_seen --network_ver v0.2.7.4.p_40_8 --camera realsense
# python eval.py --split test_seen --network_ver v0.2.7.4.p_40_10 --camera realsense
# python eval.py --split test_seen --network_ver v0.2.7.4.p_40_20 --camera realsense
# python eval.py --split test_seen --network_ver v0.2.7.4.p_40_50 --camera realsense
# python eval.py --split test_similar --network_ver v0.2.7.4.p_40_20 --camera realsense
# python eval.py --split test_novel --network_ver v0.2.7.4.p_40_20 --camera realsense
# python eval.py --split test_similar --network_ver v0.2.7.4.p_50 --camera kinect
# python eval.py --split test_novel --network_ver v0.2.7.4.p_50 --camera kinect
python eval.py --split test_seen --network_ver v0.2.7.4.p_60_20 --camera kinect
python eval.py --split test_similar --network_ver v0.2.7.4.p_60_20 --camera kinect
python eval.py --split test_novel --network_ver v0.2.7.4.p_60_20 --camera kinect