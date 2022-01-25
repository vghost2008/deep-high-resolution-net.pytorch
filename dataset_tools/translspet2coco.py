from distutils.ccompiler import new_compiler
import scipy.io as scio
from iotoolkit.mat_data import MatData
from iotoolkit.coco_toolkit import JOINTS_PAIR
import pickle
import wml_utils as wmlu
import random
import numpy as np
import cv2
import os
import os.path as osp
import img_utils as wmli
import object_detection2.keypoints as odk
import object_detection2.visualization as odv
import object_detection2.bboxes as odb
from keypoints.get_keypoints import KPDetection
from itertools import count

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
class Trans2COCO:
    def __init__(self) -> None:
        self.dst_idxs = [5,6,7,8,9,10,11,12,13,14,15,16]
        self.src_idxs = [9,8,10,7,11,6,3,2,4,1,5,0]
        self.coco_idxs = [0,1,2,3,4]

    def __call__(self,mpii_kps,coco_kps):
        if len(mpii_kps.shape)==2:
            return self.trans_one(mpii_kps,coco_kps)
        res = []
        for mp,coco in zip(mpii_kps,coco_kps):
            res.append(self.trans_one(mp,coco))
        return np.array(res)

    def trans_one(self,mpii_kps,coco_kps):
        '''
        img: [RGB]
        '''
        res = np.zeros([17,3],dtype=np.float32)
        res[self.dst_idxs] = mpii_kps[self.src_idxs]
        res[self.coco_idxs] = coco_kps[self.coco_idxs]
        return res


if __name__ == "__main__":
    mat_path = "/home/wj/ai/mldata1/lspet/lspet/joints.mat"
    images_dir = "/home/wj/ai/mldata1/lspet/lspet/images"
    save_dir = wmlu.get_unused_path("/home/wj/ai/mldata1/lspet/tmp/vis")
    coco_model = KPDetection()
    data = MatData(mat_path).data['joints']
    kps = np.transpose(data,[2,0,1])
    bboxes = odk.npbatchget_bboxes(kps)
    trans_model = Trans2COCO()
    new_coco_data = []
    for i,bbox,kp in zip(count(),bboxes,kps):
        img_name = f"im{i+1:05d}.jpg"
        file_path = osp.join(images_dir,img_name)
        img = wmli.imread(file_path)
        org_bboxes = np.array([bbox])
        max_size = img.shape[:2][::-1]
        org_bboxes = odb.npscale_bboxes(org_bboxes,1.2,max_size=max_size)
        t_bboxes = odb.npscale_bboxes(org_bboxes,1.25,max_size=max_size)
        coco_kps = coco_model.get_kps_by_bboxes(img,t_bboxes,return_fea=False)
        kp = np.array([kp])
        kps = trans_model(kp,coco_kps)
        t_bboxes = odb.npchangexyorder(t_bboxes)
        img = odv.draw_bboxes(img,bboxes=t_bboxes,is_relative_coordinate=False)
        img = odv.draw_keypoints(img,kps,no_line=False,joints_pair=JOINTS_PAIR)
        img = odv.draw_keypoints(img,coco_kps,no_line=True)
        save_path = osp.join(save_dir,img_name)
        wmli.imwrite(save_path,img)
        new_coco_data.append([img_name,org_bboxes,kps])

    coco_pt_path = "/home/wj/ai/mldata1/lspet/lspet/lspet_coco.pt"
    with open(coco_pt_path,"wb") as f:
        pickle.dump(new_coco_data,f)
    exit(0)