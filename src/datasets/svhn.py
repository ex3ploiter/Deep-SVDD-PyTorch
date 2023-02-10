from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import SVHN
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from typing import Any, Callable, Optional, Tuple
import numpy as np
import torchvision.transforms as transforms


class SVHN_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        self.min_max = [
            (-9.810971260070801, 11.722309112548828),
            (-11.558295249938965, 11.186696052551270),
            (-12.608600616455078, 12.058191299438477),
            (-11.361378669738770, 9.659506797790527),
            (-11.284592628479004, 11.021983146667480),
            (-11.390562057495117, 11.181527137756348),
            (-11.062557220458984, 9.009821891784668),
            (-13.004243850708008, 9.230701446533203),
            (-14.876477241516113, 10.149822235107422),
            (-9.741143226623535, 12.573694229125977)
        ]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),])
                                        # transforms.Lambda(
                                        #     lambda x: global_contrast_normalization(x, scale='l1')),
                                        # transforms.Normalize([min_max[normal_class][0]] * 3,
                                        #                      [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        train_set = MySVHN(root=self.root, split='train', download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.labels, self.normal_classes)
        
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MySVHN(root=self.root, split='test', download=True,
                                  transform=transform, target_transform=target_transform)


class MySVHN(SVHN):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MySVHN, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return img, target,index