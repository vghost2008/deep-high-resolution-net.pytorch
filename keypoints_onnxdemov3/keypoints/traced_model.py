import os.path as osp
import cv2
import math
import torchvision
import torch
import numpy as np


curdir_path = osp.dirname(__file__)


class YOLOXDetection:
    def __init__(self, model_path=None,conf_thre=0.1,nms_thre=0.4):
        if model_path is None:
            model_path = osp.join(curdir_path, "yolox_m.torch")

        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        print(f"Load {model_path}")
        self.model = torch.jit.load(model_path)
        print(self.model)
        self.device = torch.device("cuda:0")
        self.post_process = True

    @staticmethod
    def preproc(img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        if math.fabs(r - 1) < 0.001:
            resized_img = img
        else:
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    @staticmethod
    def demo_postprocess(outputs, img_size, p6=False):

        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides #cx,cy
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides #w,h

        return outputs

    @staticmethod
    def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def __call__(self, img):
        '''

        Args:
            img: BGR order

        Returns:
            ans: [N,7] (xmin,ymin,xmax,ymax,obj_probs,cls_probs,cls),B
        '''
        input_shape = (640,640)
        img,r = self.preproc(img,input_shape)
        img = np.expand_dims(img,axis=0)
        with torch.no_grad():
            img = torch.from_numpy(img).to(self.device)
            output = self.model(img)
        if self.post_process:
            output = self.demo_postprocess(output.cpu().to(torch.float32).numpy(), input_shape)
            output = torch.from_numpy(output)
        output = self.postprocess(output,5,conf_thre=self.conf_thre,nms_thre=self.nms_thre,class_agnostic=True)
        output = output[0]
        if output is None:
            return np.zeros([0,7],dtype=np.float32)
        output = output.cpu().detach().numpy()
        output[...,:4] = output[...,:4]/r
        bboxes = output[...,:4]
        wh = bboxes[...,2:]-bboxes[...,:2]
        wh_mask = wh>1
        size_mask = np.logical_and(wh_mask[...,0],wh_mask[...,1])
        output = output[size_mask]

        return output