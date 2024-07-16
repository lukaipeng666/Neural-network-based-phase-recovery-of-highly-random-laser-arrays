from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transform
from PIL import Image
import os




class CreateDatasets2(Dataset):
    def __init__(self, ori_imglist, output_imglist):
        self.output_imglist = output_imglist
        self.ori_imglist = ori_imglist

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = self.ori_imglist[item]
        real_img = self.output_imglist[item]
        ori_img = ori_img.unsqueeze(0)
        real_img = [real_img[0].unsqueeze(0), real_img[1].unsqueeze(0)]
        return ori_img, real_img
