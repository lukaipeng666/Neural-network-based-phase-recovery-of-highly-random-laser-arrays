from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from PIL import Image
import os


def normalize_to_minus1_1(tensor):
    return tensor * 2.0 - 1.0


class CreateDatasets(Dataset):
    def __init__(self, ori_imglist, output_path, img_size):
        self.output_path = output_path
        self.ori_imglist = ori_imglist
        self.output_images = output_path
        self.transform = transform.Compose([
            transform.Resize((img_size, img_size)),
            transform.ToTensor(),
            normalize_to_minus1_1
        ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = Image.open(self.ori_imglist[item])
        filename = os.path.basename(self.ori_imglist[item].replace('./inputs\\', os.sep))
        real_img_path = os.path.join(self.output_path, filename)
        real_img = Image.open(real_img_path)
        ori_img = self.transform(ori_img.copy())
        real_img = self.transform(real_img)
        return ori_img, real_img
