import torch

class KPDetection:
    def __init__(self,so_path,kp_pt,person_pt) -> None:
        self.so_path = so_path
        self.kp_pt = kp_pt
        self.person_pt = person_pt
        torch.ops.load_library(so_path)
        self.model = torch.ops.hr_keypoints

    def forward(self,img):
        img = torch.from_numpy(img)
        res = self.model(img,self.kp_pt,self.person_pt)
        return res.cpu().numpy()