# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
import wtorch.utils as wtu

import dataset
import models
config_w48 = {
    'cfg':'experiments/coco/hrnet/w48_384x288_adam_lr1e-3-finetune.yaml',
    #'ckpt':'/home/wj/ai/mldata1/hrnet/weights/coco_mpii/pose_hrnet/w48_384x288_adam_lr1e-3-finetune/final_state.pth',
    'ckpt':'/home/wj/ai/work/deep-high-resolution-net.pytorch/boeweights/w48_384x288_811.pth',
    'gtbboxes':True,
}
config_w32 = {
    'cfg':'experiments/coco/hrnet/w32_256x192_adam_lr1e-3-finetune.yaml',
    #'ckpt':'/home/wj/ai/mldata1/hrnet/weights/coco_mpii/pose_hrnet/w32_256x192_adam_lr1e-3-finetune/final_state.pth',
    'ckpt':'/home/wj/ai/work/deep-high-resolution-net.pytorch/boeweights/w32_256x192_790.pth',
    'gtbboxes':True,
}
config_ww32 = {
    'cfg':'experiments/coco/whrnet/w32_256x192_adam_lr1e-3.yaml',
    'ckpt':'/home/wj/ai/mldata1/hrnet/weights/coco_mpii/pose_whrnet/w32_256x192_adam_lr1e-3/final_state.pth',
    'gtbboxes':True,
}

test_config = config_w32

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=test_config['cfg'],
                        required=False,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.GPUS = (0,1,2,3)
    cfg.GPUS = (0,)
    cfg.TEST.FLIP_TEST = False
    cfg.DEBUG.SAVE_BATCH_IMAGES_GT = False
    cfg.DEBUG.SAVE_HEATMAPS_GT = False
    cfg.DEBUG.SAVE_HEATMAPS_PRED = False
    cfg.TEST.MODEL_FILE = test_config['ckpt']
    cfg.TEST.USE_GT_BBOX = test_config['gtbboxes']
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid_test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        data = torch.load(cfg.TEST.MODEL_FILE)
        if 'state_dict' in data:
            data = data['state_dict']
        data = wtu.remove_prefix_from_state_dict(data,prefix="module.")
        model.load_state_dict(data, strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
