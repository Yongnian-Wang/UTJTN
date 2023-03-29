import torch
import torch.nn as nn
from torchvision import transforms
class BatchToTensor(nn.Module):
    def __init__(self):
        super(BatchToTensor, self).__init__()
    def forward(self, imgs):
        # transforms.ToTensor() 把图片数据转换成tensor数据类型
        return [transforms.ToTensor()(img) for img in imgs]

class YCbCrToRGB(nn.Module):
    def __init__(self):
        super(YCbCrToRGB, self).__init__()
    def forward(self, img):
        return torch.stack((img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
                            img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 - (img[:, 2, :, :] - 128 / 256.) * 0.714136,
                            img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.) * 1.772),
                            dim=1)
class RGBToYCbCr(nn.Module):
    def __init__(self):
        super(RGBToYCbCr, self).__init__()
    def forward(self, img):
        return torch.stack((0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
                           128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
                           128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
                          dim=1)