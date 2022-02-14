from iotoolkit.aic_keypoint import *
import wml_utils as wmlu
import sys
import os.path as osp
import img_utils as wmli
import numpy as np
import object_detection2.visualization as odv
import pickle
from iotoolkit.coco_toolkit import JOINTS_PAIR
import object_detection2.bboxes as odb
from keypoints.get_keypoints import KPDetection
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == "__main__":
    root_dir = '/home/wj/ai/mldata/aic'
    file_path = [('ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json','ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'),
    ('ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json','ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902')]
    img_dir_path = root_dir
    save_dir = '/home/wj/ai/mldata/aic/tmp/vis'
    save_dir = wmlu.get_unused_path(save_dir)
    print("save dir: ",save_dir)
    coco_model = KPDetection()
    trans_model = Trans2COCO()

    datas = []
    for path,sub_dir in file_path:
        path = osp.join(root_dir,path)
        data_t = read_aic_keypoint(path,sub_dir)
        datas.extend(data_t)
    #datas = datas[:200]
    do_vis = False
    new_coco_data = []
    KPDetection.threshold = 0.3
    #random.seed(21)
    #random.shuffle(datas)
    for i,data in enumerate(datas):
        sys.stdout.write(f"\r{i}/{len(datas)}")
        img_name,bboxes,kps = data
        file_path = osp.join(img_dir_path,img_name)
        vis = (kps[...,-1]>0).astype(np.int32)
        if np.sum(vis)<3:
                print(f"Skip {img_name}\n\n")
        npz_save_path = wmlu.change_suffix(file_path,"npz")    
        if osp.exists(npz_save_path):
            with open(npz_save_path,"rb") as f:
                coco_kps = pickle.load(f)
            print(f"Load coco keypoints ",coco_kps)
        else:
            img = wmli.imread(file_path)
            org_bboxes = np.array(bboxes)
            max_size = img.shape[:2][::-1]
            t_bboxes = odb.npscale_bboxes(org_bboxes,1.2,max_size=max_size)
            coco_kps = coco_model.get_kps_by_bboxes(img,t_bboxes,return_fea=False)
            with open(npz_save_path,"wb") as f:
                pickle.dump(coco_kps,f)
        kps = trans_model(kps,coco_kps)
        if do_vis:
            file_path = osp.join(img_dir_path,img_name)
            img = wmli.imread(file_path)
            t_bboxes = odb.npchangexyorder(org_bboxes)
            img = odv.draw_bboxes(img,bboxes=t_bboxes,is_relative_coordinate=False)
            img = odv.draw_keypoints(img,kps,no_line=False,joints_pair=JOINTS_PAIR)
            save_path = osp.join(save_dir,img_name)
            wmli.imwrite(save_path,img)
        new_coco_data.append([img_name,org_bboxes,kps])

    coco_pt_path = osp.join(root_dir,'aic_cocov2.pt')
    with open(coco_pt_path,"wb") as f:
        pickle.dump(new_coco_data,f)
    exit(0)