import sys
import os.path as osp
from vis_utils import *
sys.path.append(osp.dirname(osp.dirname(__file__)))
from demo_toolkit import *
import os
import numpy as np
from keypoints_cpp import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pdir_path = osp.dirname(osp.dirname(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

class Model:
    def __init__(self):
        self.model = KPDetection("/home/wj/ai/work/chunxi_bank_ai/higher_hrnet_cpp/libhrnetv2/build/libhrnet.so",
        "/home/wj/ai/work/deep-high-resolution-net.pytorch/boeweights/keypoints.torch",
        "/home/wj/ai/work/deep-high-resolution-net.pytorch/boeweights/person_m.torch")

    def __call__(self, img):
        raw_img = img
        with TimeThis():
            ans = self.model(img)
        img = show_keypoints(raw_img,ans,threshold=0.2)
        return img

if __name__ == "__main__":
    vd = VideoDemo(Model(),save_path="tmp3.mp4",buffer_size=1,show_video=False,max_frame_cn=1000)
    #video_path = "/home/wj/ai/mldata/boeoffice/test_data/test3.webm"
    print(f"DATE: 2021-11-29")
    '''video_path = "/home/wj/ai/mldata/global_traj/tennis1.mp4"
    video_path = "/home/wj/ai/mldata/pose3d/basketball2.mp4"
    video_path = "/home/wj/ai/mldata/keypoints_data/model_test_video/railway_station_scene.avi"'''
    video_path = "/home/wj/ai/mldata/keypoints_data/model_test_video/cto_staff_scene.avi"
    #video_path = None
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    vd.inference_loop(video_path)
    vd.close()
