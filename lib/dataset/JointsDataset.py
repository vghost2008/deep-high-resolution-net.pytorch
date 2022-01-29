# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import object_detection2.keypoints as odk
import copy
import logging
import random
import object_detection2.bboxes as odb
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import object_detection2.visualization as odv
from iotoolkit.coco_toolkit import JOINTS_PAIR as COCO_JOINTS_PAIR
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
import wtorch.dataset_toolkit as tdt
import img_utils as wmli
from datadef import *


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        '''
        get half body bbox
        '''
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        cur_db = self.db[idx]
        if isinstance(cur_db,tdt.DataUnit):
            cur_db = cur_db.sample()
        db_rec = copy.deepcopy(cur_db)

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            try:
                if np.random.rand()<0.4:
                    return self.trans_data_type0(data_numpy,db_rec)
            except Exception as e:
                print(f"Trans data error {e}")
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
        else:
            return self.trans_data_type0(data_numpy,db_rec)
            pass

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        try:
            target, target_weight = self.generate_target(joints, joints_vis)
        except Exception as e:
            print(db_rec)
            print(joints,joints_vis)
            print(e)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta
    
    @staticmethod
    def save_vis_kps(img,kps,path,bbox=None):
        img = odv.draw_keypoints(img,kps[...,:3],joints_pair=COCO_JOINTS_PAIR)
        if bbox is not None:
            img = odv.draw_bbox(img,bbox,xy_order=True)
        wmli.imwrite(path,img)

    def trans_data_type0(self,data_numpy,db_rec):
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        bbox = db_rec.get('clean_bbox',None)
        if bbox is None:
            bbox = odk.npget_bbox(joints)
        bbox = odb.npscale_bboxes(bbox,1.25,max_size=data_numpy.shape[:2][::-1])
        score = db_rec['score'] if 'score' in db_rec else 1
        c = db_rec['center']
        
        #s = 1.4
        #r = 45
        #self.save_vis_kps(data_numpy,joints,"a.jpg",bbox)
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s =  np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
            data_numpy,joints,bbox = odk.rotate(r,data_numpy,joints,bbox,s)
            joints_vis[:,0] = joints[:,2] 
            joints_vis[:,1] = joints[:,2] 

            #self.save_vis_kps(data_numpy,joints,"b.jpg",bbox)
            if self.flip and random.random() <= 0.5:
                data_numpy,joints,joints_vis,bbox = odk.flip(data_numpy,joints,joints_vis,self.flip_pairs,
                                                        bbox=bbox)
        else:
            r = 0.0
        #self.save_vis_kps(data_numpy,joints,"b1.jpg",bbox)
        #bbox = odk.npget_bbox(joints)
        #bbox = odb.npscale_bboxes(bbox,1.4)
        #org_img = data_numpy
        data_numpy,bbox = self.cut_and_resize(data_numpy,[bbox],size=self.image_size)
        data_numpy = data_numpy[0]
        bbox = bbox[0]
        '''img_a = wmli.sub_imagev2(org_img,bbox.astype(np.int32))
        img_a = wmli.resize_img(img_a,self.image_size)
        img_b = data_numpy
        wmli.imwrite("a.jpg",img_a)
        wmli.imwrite("b.jpg",img_b)'''

        c = np.array([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2],dtype=np.float32)
        scale = np.array(
            [
                (bbox[2]-bbox[0]) * 1.0 / self.pixel_std,
                (bbox[3]-bbox[1]) * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        joints = odk.cut2size(joints,bbox,self.image_size)
        #self.save_vis_kps(data_numpy,joints,"c.jpg",bbox)

        target, target_weight = self.generate_target(joints, joints_vis)
        if np.max(target_weight)<0.1:
            #print(f"ERROR")
            pass
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        if self.transform:
            data_numpy = self.transform(data_numpy)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': scale,
            'rotation': r,
            'score': score
        }

        return data_numpy, target, target_weight, meta

    @staticmethod
    def cut_and_resize(img,bboxes,size=(288,384)):
        res = []
        res_bboxes = []
        bboxes = np.array(bboxes).astype(np.int32)
        for i,bbox in enumerate(bboxes):
            cur_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
            if cur_img.shape[0]>1 and cur_img.shape[1]>1:
                #cur_img = cv2.resize(cur_img,size,interpolation=cv2.INTER_LINEAR)
                cur_img,bbox = JointsDataset.resize_img(cur_img,bbox,size)
            else:
                cur_img = np.zeros([size[1],size[0],3],dtype=np.float32)
            res.append(cur_img)
            res_bboxes.append(bbox)
        return res,np.array(res_bboxes,dtype=np.float32)
    
    @staticmethod
    def resize_img(img,bbox,target_size,pad_color=(127,127,127)):
        res = np.ndarray([target_size[1],target_size[0],3],dtype=np.uint8)
        res[:,:] = np.array(pad_color,dtype=np.uint8)
        ratio = target_size[0]/target_size[1]
        bbox_cx = (bbox[2]+bbox[0])/2
        bbox_cy = (bbox[3]+bbox[1])/2
        bbox_w = (bbox[2]-bbox[0])
        bbox_h = (bbox[3]-bbox[1])
        if img.shape[1]>ratio*img.shape[0]:
            nw = target_size[0]
            nh = int(target_size[0]*img.shape[0]/img.shape[1])
            bbox_h = bbox_w/ratio
        else:
            nh = target_size[1]
            nw = int(target_size[1]*img.shape[1]/img.shape[0])
            bbox_w = bbox_h*ratio
    
        img = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
        xoffset = (target_size[0]-nw)//2
        yoffset = (target_size[1]-nh)//2
        res[yoffset:yoffset+nh,xoffset:xoffset+nw] = img
        bbox = np.array([bbox_cx-bbox_w/2,bbox_cy-bbox_h/2,bbox_cx+bbox_w/2,bbox_cy+bbox_h/2],dtype=np.float32)
        return res,bbox

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        joints_vis = np.minimum(joints_vis,1)
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3 #sigma=3

            for joint_id in range(self.num_joints):
                if target_weight[joint_id,0]<0.1:
                    continue
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
