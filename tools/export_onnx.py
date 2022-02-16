import argparse
import _init_paths
import os
from loguru import logger
import onnx
from onnxsim import simplify
import models
import torch
from config import cfg
from config import update_config
import os.path as osp


from torch import nn

config1 = {
    'cfg':'experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml',
    'ckpt':"weights/pose_hrnet_w48_384x288.pth",
    'input_shape':[1, 3,384, 288],
    }

config2 = {
    'cfg':'experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
    'ckpt':"boeweights/w32_256x192_790.pth",
    'input_shape':[1, 3,256, 192],
    }
config3 = {
    'cfg':'experiments/coco/whrnet/w32_384x256_adam_lr1e-3-finetune.yaml',
    'ckpt':"",
    'input_shape':[1, 3,384, 288],
    }
export_config = config3
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=export_config['cfg'])
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam',action='store_true')
    parser.add_argument('--image',type=str)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--showFps',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

@logger.catch
def main():
    device = torch.device("cuda:0")
    args = parse_args()
    update_config(cfg, args)

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    pose_model.to(device)
    ckpt = export_config['ckpt']
    if ckpt is not None and ckpt != "" and osp.exists(ckpt):
        state_dict = torch.load(ckpt)
        pose_model.load_state_dict(state_dict, strict=False)
        logger.info(f"loading checkpoint {ckpt} done.")
    else:
        print(f"===============================================================")
        print(f"Find ckpt {ckpt} faild.")
        print(f"===============================================================")
    pose_model.eval()
    pose_model.add_preprocess = True


    #dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    #dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    #dummy_input = torch.randn(1, 3, 384, 640)
    dummy_input = torch.randn(*export_config['input_shape'])
    dummy_input = dummy_input.to(device)
    output = "output"
    input = "input"
    output_path = "output.onnx"
    dynamic = True

    torch.onnx._export(
        pose_model,
        dummy_input,
        output_path,
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={"input_0": {0: 'B'},
                      "output_0": {0: 'B'}} ,
        opset_version=11,
        
    )
    logger.info("generated onnx model named {}".format(output_path))


    input_shapes =  list(dummy_input.shape)
        
    onnx_model = onnx.load(output_path)
    model_simp, check = simplify(onnx_model,
                                 dynamic_input_shape=dynamic,
                                 input_shapes={'input_0':input_shapes})
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    logger.info("generated simplified onnx model named {}".format(output_path))

if __name__ == "__main__":
    main()
