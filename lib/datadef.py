import os.path as osp
import os

LEFT_NR = 2
RIGHT_NR = 2

def get_data_dir(sub_path=None,create_dir=False):
    data_dir = osp.expanduser("~/ai/mldata1/hrnet")

    if sub_path is not None:
        data_dir = osp.join(data_dir,sub_path)

    if create_dir and not osp.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir

ckpt_dir = get_data_dir("weights")


def get_ckpt_dir():
    ckpt_dir = get_data_dir("weights")
    return ckpt_dir

def get_log_dir(suffix=""):
    torch_loger_dir = get_data_dir("tmp/tbloger"+suffix)
    return torch_loger_dir


def is_debug(default_value = True):
    print(f"is_debug {default_value}")
    return default_value