from iotoolkit.crowd_pose import *
from keypoints.get_keypoints import KPDetection
from itertools import count
import img_utils as wmli
import object_detection2.visualization as odv
import wml_utils as wmlu
import pickle
from iotoolkit.coco_toolkit import JOINTS_PAIR
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"



if __name__ == "__main__":
    file_path = ['/home/wj/ai/mldata1/crowd_pose/CrowdPose/crowdpose_train.json',
        '/home/wj/ai/mldata1/crowd_pose/CrowdPose/crowdpose_val.json',
        '/home/wj/ai/mldata1/crowd_pose/CrowdPose/crowdpose_test.json']
    img_dir_path = '/home/wj/ai/mldata1/crowd_pose/images'
    save_dir = '/home/wj/ai/mldata1/crowd_pose/tmp/vis'
    save_dir = wmlu.get_unused_path(save_dir)
    trans_model = Trans2COCO()
    coco_model = KPDetection()

    datas = []
    for path in file_path:
        data_t = read_crowd_pose(path)
        datas.extend(data_t)
    do_vis = True
    new_coco_data = []
    for i,data in enumerate(datas):
        sys.stdout.write(f"\r{i}/{len(datas)}")
        img_name,kps,bboxes = data
        file_path = osp.join(img_dir_path,img_name)
        vis = kps[...,-1]
        if np.sum(vis)<0.5:
                print(f"Skip {img_name}\n\n")
        img = wmli.imread(file_path)
        org_bboxes = np.array(bboxes)
        max_size = img.shape[:2][::-1]
        #org_bboxes = odb.npscale_bboxes(org_bboxes,1.2,max_size=max_size)
        t_bboxes = odb.npscale_bboxes(org_bboxes,1.25,max_size=max_size)
        coco_kps = coco_model.get_kps_by_bboxes(img,t_bboxes,return_fea=False)
        kps = trans_model(kps,coco_kps)
        if do_vis:
            t_bboxes = odb.npchangexyorder(t_bboxes)
            img = odv.draw_bboxes(img,bboxes=t_bboxes,is_relative_coordinate=False)
            img = odv.draw_keypoints(img,kps,no_line=False,joints_pair=JOINTS_PAIR)
            img = odv.draw_keypoints(img,coco_kps,no_line=True)
            save_path = osp.join(save_dir,img_name)
            wmli.imwrite(save_path,img)
        new_coco_data.append([img_name,org_bboxes,kps])

    coco_pt_path = '/home/wj/ai/mldata1/crowd_pose/CrowdPose/crowdpose_coco.pt',
    with open(coco_pt_path,"wb") as f:
        pickle.dump(new_coco_data,f)
    exit(0)
