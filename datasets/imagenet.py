from torchvision import transforms,datasets 


from torch.utils.data import Dataset
import torch

def ImageNet(config,split):
    t = transforms.Compose([transforms.RandomOrder(
                            [transforms.RandomVerticalFlip(),
                            transforms.ColorJitter(saturation=0.75, hue=0.1),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(7)]),
                            transforms.RandomResizedCrop(512),
                            transforms.ToTensor()
    ])
    return datasets.ImageNet(config['root'], split=split,transform = t)
