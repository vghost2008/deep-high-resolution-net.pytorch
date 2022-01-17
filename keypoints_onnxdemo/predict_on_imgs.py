import sys
import os.path as osp
from vis_utils import *
sys.path.append(osp.dirname(osp.dirname(__file__)))
from demo_toolkit import *
import os
import numpy as np
from keypoints.get_keypoints import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pdir_path = osp.dirname(osp.dirname(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == "__main__":
    data_path = "/home/wj/ai/mldata/keypoints_data/test_imgs"
    files = glob.glob(osp.join(data_path,"*.png"))
    save_dir = "/home/wj/ai/mldata/0day/tmp/kps"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    model = KPDetection()
    for file in files:
        img = cv2.imread(file)
        img = img[...,::-1]
        ans = model(img)
        img = show_keypoints(img,ans,threshold=0.2)
        save_path = osp.join(save_dir,osp.basename(file))
        cv2.imwrite(save_path,img)
