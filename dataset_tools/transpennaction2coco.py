from iotoolkit.penn_action import *
from keypoints.get_keypoints import KPDetection
from itertools import count
import img_utils as wmli
import object_detection2.visualization as odv
import wml_utils as wmlu
import pickle
from iotoolkit.coco_toolkit import JOINTS_PAIR
import sys


if __name__ == "__main__":
    dir_path = "/home/wj/ai/mldata1/penn_action/Penn_Action/labels"
    img_dir_path = "/home/wj/ai/mldata1/penn_action/Penn_Action/frames"
    save_dir = wmlu.get_unused_path("/home/wj/ai/mldata1/penn_action/tmp/vis")
    trans_model = Trans2COCO()
    coco_model = KPDetection()

    datas = read_penn_action_data(dir_path)
    do_vis = False
    new_coco_data = []
    for i,data in enumerate(datas):
        sys.stdout.write(f"\r{i}/{len(datas)}")
        file,kps,bboxes = data
        bf_name = wmlu.base_name(file)
        for i,kp,bbox in zip(count(),kps,bboxes):
            img_name = f"{bf_name}/{i+1:06d}.jpg"
            file_path = osp.join(img_dir_path,img_name)
            vis = kp[...,-1]
            if np.sum(vis)<0.5:
                print(f"Skip {img_name}\n\n")
            img = wmli.imread(file_path)
            org_bboxes = np.array([bbox])
            max_size = img.shape[:2][::-1]
            org_bboxes = odb.npscale_bboxes(org_bboxes,1.2,max_size=max_size)
            t_bboxes = odb.npscale_bboxes(org_bboxes,1.25,max_size=max_size)
            coco_kps = coco_model.get_kps_by_bboxes(img,t_bboxes,return_fea=False)
            kp = np.array([kp])
            kps = trans_model(kp,coco_kps)
            if do_vis:
                t_bboxes = odb.npchangexyorder(t_bboxes)
                img = odv.draw_bboxes(img,bboxes=t_bboxes,is_relative_coordinate=False)
                img = odv.draw_keypoints(img,kps,no_line=False,joints_pair=JOINTS_PAIR)
                img = odv.draw_keypoints(img,coco_kps,no_line=True)
                save_path = osp.join(save_dir,img_name)
                wmli.imwrite(save_path,img)
            new_coco_data.append([img_name,org_bboxes,kps])

    coco_pt_path = "/home/wj/ai/mldata1/penn_action/Penn_Action/penn_action_coco.pt"
    with open(coco_pt_path,"wb") as f:
        pickle.dump(new_coco_data,f)
    exit(0)
