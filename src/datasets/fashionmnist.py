from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms


class FashionMNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        self.min_max = [(-2.681241989135742, 24.854305267333984),
                   (-2.577857017517090, 11.169789314270020),
                   (-2.808170318603516, 19.133543014526367),
                   (-1.953365325927734, 18.656726837158203),
                   (-2.610385417938232, 19.166683197021484),
                   (-1.235852122306824, 28.463108062744141),
                   (-3.251605987548828, 24.196832656860352),
                   (-1.081444144248962, 16.878818511962891),
                   (-3.656097888946533, 11.350274085998535),
                   (-1.385928869247437, 11.426652908325195)]

        # FashionMNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),])
                                        # transforms.Lambda(
                                        #     lambda x: global_contrast_normalization(x, scale='l1')),
                                        # transforms.Normalize([min_max[normal_class][0]],
                                        #                      [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        train_set = MyFashionMNIST(root=self.root, train=True, download=True,
                                   transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(
            train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyFashionMNIST(root=self.root, train=False, download=True,
                                       transform=transform, target_transform=target_transform)


class MyFashionMNIST(FashionMNIST):
    """Torchvision FashionMNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyFashionMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the FashionMNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed