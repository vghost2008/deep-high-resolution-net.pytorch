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


from torch import nn

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml')
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
                return self.model(x)

@logger.catch
def main():
    device = torch.device("cuda:0")
    args = parse_args()
    update_config(cfg, args)

    pose_model = eval('models.'+"pose_hrnet"+'.get_pose_net')(
        cfg, is_train=False
    )
    pose_model.to(device)
    pose_model.load_state_dict(torch.load("weights/pose_hrnet_w48_384x288.pth"), strict=False)
    pose_model.eval()

    logger.info("loading checkpoint done.")
    #dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    #dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    #dummy_input = torch.randn(1, 3, 384, 640)
    input = torch.randn(1, 3, 384, 288)
    input = input.to(device)
    output_path = "output.torch"
    logger.info("generated onnx model named {}".format(output_path))
    device = torch.device("cuda")
    pose_model = TraceWrape(pose_model)
    traced_model = torch.jit.trace(pose_model, input)
    print(traced_model.code)
    traced_model.save(output_path)
    v = traced_model(input)
    print(f"Save path {output_path}")

if __name__ == "__main__":
    main()