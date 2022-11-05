import glob
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, root_path, transform=None, transform2=None):
        """ Initialization """
        self.list_png = sorted(glob.glob(root_path + '/images/I/*.png'))
        self.transform = transform
        self.transform2 = transform2


    def __getitem__(self, index):
        img_path = self.list_png[index]
        IHD_path = img_path.replace('I', 'IHD')
        img = Image.open(img_path)
        IHD_image = Image.open(IHD_path)

        sample = {'image': img, 'ground_truth': IHD_image}
        if self.transform:
            sample["image"] = self.transform(img)
            sample["ground_truth"] =  self.transform2(IHD_image)

        return sample

    def __len__(self):
        """ Denotes the toal number of samples """
        return len(self.list_png)
        