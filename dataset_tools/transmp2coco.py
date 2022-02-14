from distutils.ccompiler import new_compiler
import scipy.io as scio
from iotoolkit.mat_data import MatData
from iotoolkit.mpii import *
from iotoolkit.coco_toolkit import JOINTS_PAIR
import pickle
import wml_utils as wmlu
import random
import cv2
import os
import os.path as osp
import img_utils as wmli
import object_detection2.keypoints as odk
import object_detection2.visualization as odv
import object_detection2.bboxes as odb
from keypoints.get_keypoints import KPDetection

os.environ['CUDA_VISIBLE_DEVICES'] = "3"



if __name__ == "__main__":
    mat_path = "/home/wj/ai/mldata1/MPII/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
    pt_path = "/home/wj/ai/mldata1/MPII/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.pt"
    coco_pt_path = "/home/wj/ai/mldata1/MPII/MPII/mpii_human_pose_v1_u12_2/mpii_cocov2.pt"
    imgs_dir = "/home/wj/ai/mldata1/MPII/MPII/images"
    save_dir = wmlu.get_unused_path("/home/wj/ai/mldata1/MPII/tmp/vis")
    print(f"save dir ",save_dir)
    coco_model = KPDetection()
    trans_model = Trans2COCO()
    '''mpii_data = read_mpii_data(mat_path)
    with open(pt_path,"wb") as f:
        pickle.dump(mpii_data,f)'''
    with open(pt_path,"rb") as f:
        mpii_data = pickle.load(f)
    new_coco_data = []
    do_vis = False
    for data in mpii_data:
        img_name,bboxes,kps,head_bboxes = data
        org_bboxes = bboxes
        org_bboxes1 = bboxes
        img_path = osp.join(imgs_dir,img_name)
        img = wmli.imread(img_path)
        max_size = img.shape[:2][::-1]
        bboxes = odb.npscale_bboxes(org_bboxes,1.25,max_size=max_size)
        coco_kps = coco_model.get_kps_by_bboxes(img,bboxes,return_fea=False)
        kps = trans_model(kps,coco_kps,head_bboxes)
        kps_bboxes = odk.npbatchget_bboxes(kps)
        new_bboxes = []
        for i in range(kps_bboxes.shape[0]):
            kp_bbox = kps_bboxes[i]
            bbox = org_bboxes[i]
            if kp_bbox is not None:
                bbox = odb.bbox_of_boxes([kp_bbox,org_bboxes[i]])
            new_bboxes.append(bbox)
        org_bboxes = np.array(new_bboxes)
        new_coco_data.append([img_name,org_bboxes,kps])
        if do_vis:
            t_bboxes = odb.npchangexyorder(org_bboxes)
            #t_bboxes = odb.npchangexyorder(head_bboxes)
            img = odv.draw_bboxes(img,bboxes=t_bboxes,is_relative_coordinate=False)
            #img = odv.draw_keypoints(img,kps,no_line=False,joints_pair=JOINTS_PAIR)
            img = odv.draw_keypoints(img,kps,no_line=False,joints_pair=JOINTS_PAIR)
            #img = odv.draw_keypoints(img,coco_kps,no_line=True)
            save_path = osp.join(save_dir,img_name)
            wmli.imwrite(save_path,img)
    
    with open(coco_pt_path,"wb") as f:
        pickle.dump(new_coco_data,f)

    '''for data in mpii_data:
        img_name,bboxes,kps = data
        bboxes = odb.npscale_bboxes(bboxes,1.25)
        img_path = osp.join(imgs_dir,img_name)
        img = wmli.imread(img_path)
        coco_kps = coco_model.get_kps_by_bboxes(img,bboxes,return_fea=False)
        kps = trans_model(kps,coco_kps)
        bboxes = odb.npchangexyorder(bboxes)
        img = odv.draw_bboxes(img,bboxes=bboxes,is_relative_coordinate=False)
        img = odv.draw_keypoints(img,kps,no_line=False,joints_pair=JOINTS_PAIR)
        img = odv.draw_keypoints(img,coco_kps,no_line=True)
        save_path = osp.join(save_dir,img_name)

        wmli.imwrite(save_path,img)'''



    print(mpii_data)
