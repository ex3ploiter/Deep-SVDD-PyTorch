from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms

import glob 
import os

class MVTec_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 15))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        self.min_max = [(-1.280467867851257 , 1.380750298500061),
(-2.348408460617065 , 4.638308525085449),
(-3.146009206771851 , 1.405009150505066),
(-3.430535793304443 , 6.451485157012939),
(-2.849516868591309 , 3.375417232513428),
(-3.163123607635498 , 13.883568763732910),
(-2.112395763397217 , 7.129158973693848),
(-1.277425050735474 , 6.942065715789795),
(-0.968891263008118 , 2.238602399826050),
(-7.060428142547607 , 2.036980152130127),
(-3.211194038391113 , 6.290756702423096),
(-0.834994316101074 , 4.577285289764404),
(-1.986122012138367 , 5.502355098724365),
(-2.754846811294556 , 4.641113758087158),
(-1.266530275344849 , 2.525079250335693)]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([
          transforms.Resize(256),
          transforms.ToTensor(),
                                        # transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        # transforms.Normalize([min_max[normal_class][0]] * 3,
                                        #                      [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])
        ])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = MyMVTec(root=self.root, train=True,normal_class=normal_class,
                              transform=transform, target_transform=target_transform)


        self.test_set = MyMVTec(root=self.root, train=False,normal_class=normal_class,
                              transform=transform, target_transform=target_transform)






class MyMVTec(TorchvisionDataset):
    def __init__(self, root, normal_class, transform=None, target_transform=None, train=True):
        self.transform = transform
        root=os.path.join(root,'mvtec_anomaly_detection')
        
        mvtec_labels=['bottle' , 'cable' , 'capsule' , 'carpet' ,'grid' , 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood','zipper']
        category=mvtec_labels[normal_class]

        if train:
            self.image_files = glob.glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob.glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob.glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          
        #   if normal:
        #     self.image_files = normal_image_files
        #   else:
        #     self.image_files = anomaly_image_files

          self.image_files = normal_image_files+anomaly_image_files

        # self.train = train
        

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        
        return image, target, index
        

    def __len__(self):
        return len(self.image_files)
    
    