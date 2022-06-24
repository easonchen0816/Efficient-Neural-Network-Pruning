from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from config import cfg

from PIL import Image

import os
import os.path
#import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Caltech(VisionDataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.classes = sorted(os.listdir(os.path.join(self.root, "")))
        # Filter out the lines which start with 'BACKGROUND_Google' as asked in the homework
        
        self.elements = list()
        if split == 'train':
            for i in range(len(self.classes)):
                imgdirlist = list()
                ele = os.listdir(os.path.join(self.root, self.classes[i]))
                for j in range(60):
                    imgdirlist.append(os.path.join(self.root, self.classes[i], ele[j]))
                self.elements = self.elements + imgdirlist
        elif split == 'test':
            for i in range(len(self.classes)):
                imgdirlist = list()
                ele = os.listdir(os.path.join(self.root, self.classes[i]))
                for j in range(60, len(ele)):
                    imgdirlist.append(os.path.join(self.root, self.classes[i], ele[j]))
                self.elements = self.elements + imgdirlist

        else:
            return 'error : not train or test data'

    def __getitem__(self, index):
        ''' 
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        #img = Image.open(os.path.join(self.root, self.elements[index].rstrip()))
        img = pil_loader(self.elements[index].rstrip())

        target = self.classes.index(self.elements[index].rstrip().split('/')[2])

        image, label = img, target # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        # Provides a way to get the length (number of elements) of the dataset
        length =  len(self.elements)
        return length

def get_caltech():
    DATA_DIR = 'data/256_ObjectCategories'


    train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    eval_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    # 1 - Data preparation
    myTrainDS = Caltech(DATA_DIR, split = 'train', transform=train_transform)
    myTestDS = Caltech(DATA_DIR, split = 'test', transform=eval_transform)

    print('Train DS: {}'.format(len(myTrainDS)))
    print('Test DS: {}'.format(len(myTestDS)))

    # 1 - Data preparation
    myTrain_dataloader = DataLoader(myTrainDS, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle, num_workers=cfg.data.num_workers, drop_last=True)
    myTest_dataloader = DataLoader(myTestDS, batch_size=cfg.data.test_batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    return myTrain_dataloader, myTest_dataloader