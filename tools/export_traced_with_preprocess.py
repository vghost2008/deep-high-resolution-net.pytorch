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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from torch import nn

config1 = {
    'cfg':'experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml',
    'ckpt':"weights/pose_hrnet_w48_384x288.pth",
    'input_shape':[1, 3,384, 288],
    }

config2 = {
    'cfg':'experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
    'ckpt':"weights/pose_hrnet_w32_256x192.pth",
    'input_shape':[1, 3,256, 192],
    }
config11 = {
    'cfg':'experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml',
    'ckpt':"boeweights/w48_384x288_811.pth",
    'input_shape':[1, 3,384, 288],
    }

config21 = {
    'cfg':'experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
    'ckpt':"boeweights/w32_256x192_790.pth",
    'input_shape':[1, 3,256, 192],
    }
config22 = {
    'cfg':'experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
    'ckpt':"boeweights/keypoints787_256x192.pth",
    'input_shape':[1, 3,256, 192],
    }
config3 = {
    'cfg':'experiments/coco/whrnet/w32_384x256_adam_lr1e-3-finetune.yaml',
    'ckpt':"",
    'input_shape':[1, 3,384, 288],
    }

export_config = config22

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
class TraceWrape(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x = self.model(x)
        x = x.to(torch.float32)
        return x
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
    if ckpt != "" and os.path.exists(ckpt):
        state_dict = torch.load(export_config['ckpt'])
        print(f"Load {export_config['ckpt']}")
        pose_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Find {ckpt} faild.")
    pose_model.eval()

    logger.info("loading checkpoint done.")
    #dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    #dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    #dummy_input = torch.randn(1, 3, 384, 640)
    input = torch.randn(*export_config['input_shape'])
    input = input.to(device)
    output_path = "output.torch"
    logger.info("generated onnx model named {}".format(output_path))
    device = torch.device("cuda")
    pose_model.add_preprocess = True
    #pose_model = TraceWrape(pose_model)
    with torch.no_grad():
        traced_model = torch.jit.trace(pose_model, input)
    print(traced_model.code)
    print(f"Save path {output_path}")
    traced_model.save(output_path)
    v = traced_model(input)
    print(v.shape)

if __name__ == "__main__":
    main()