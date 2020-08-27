from torch.utils.data import Dataset
import numpy as np
import os 
from glob import glob
from datasets.semantickitti_conf import *
import torch

class SemanticKitti(Dataset):
    
    LABELS = LABELS
    COLOR_MAP = COLOR_MAP
    CONTENT = CONTENT
    LEARNING_MAP = LEARNING_MAP
    LEARNING_MAP_INV = LEARNING_MAP_INV
    LEARNING_IGNORE = LEARNING_IGNORE
    SPLIT = SPLIT
    AVAILABLE_TRANSFORMS = AVAILABLE_TRANSFORMS
    SENSOR_CONFIG = SENSOR_CONFIG
    IMG_PROP = IMG_PROP
    
    def __init__(self, config, split="train"):
        super().__init__()
        self.root = config["root"]
        self._normalise = config["normalise"]
        self._populate_colormap()

        if split not in list(self.SPLIT.keys()):
            raise ValueError(f"split {split} does not exist. choose from train, val or test")
        else:
            self._load_paths(split)
        
        if config["transform"] not in self.AVAILABLE_TRANSFORMS:
            raise ValueError(f"transform does not exist.")
        else:
            self._transform = self.AVAILABLE_TRANSFORMS[config["transform"]](config["img_H"], config["img_W"], self.SENSOR_CONFIG["fov_up"], self.SENSOR_CONFIG["fov_down"], normalise=config["normalise"], means=self.IMG_PROP["img_means"], std=self.IMG_PROP["img_stds"])
                
    def _load_paths(self, split):
        self._scan_paths = []
        self._label_path = []
        seqs = self.SPLIT[split]
        for seq in seqs:
            self._scan_paths.extend(sorted(glob(os.path.join(self.root, "{0:02d}".format(int(seq)), "velodyne", '*.bin'))))
            self._label_path.extend(sorted(glob(os.path.join(self.root, "{0:02d}".format(int(seq)), "labels", '*.label'))))
        if len(self._scan_paths) != len(self._label_path):
            raise ValueError((f"number of scans {len(self._scan_paths)} not equal to number of labels {len(self._label_path)}"))     

    def _populate_colormap(self):
        self.CMAP = np.zeros((max(list(self.LEARNING_MAP.keys()))+1, 3), dtype=np.float32)
        for key, value in self.COLOR_MAP.items():
            value = [value[i] for i in [2,1,0]]
            self.CMAP[key] = np.array(value, np.float32) / 255.0
    
    @staticmethod
    def open_scan(file_name):
        """
        open scan file - [x, y, z, remissions]
        returns point coordinates, remissions
        """
        scan = np.fromfile(file_name, dtype=np.float32)
        scan = scan.reshape((-1, 4)) # just for the sake of it
        return scan
    
    @staticmethod
    def open_label(file_name):
        """
        open label file - [semantic label(first 16 bits), instance label(last 16 bits)]
        returns semantic label, instance label
        """
        label = np.fromfile(file_name, dtype=np.uint32)
        label = label.reshape((-1)) # again for the sake of it
        semantic_label = label & 0xFFFF
        instance_label = label >> 16
        return semantic_label, instance_label

    def __getitem__(self, idx):
        scan = SemanticKitti.open_scan(self._scan_paths[idx])
        label = None
        if self._label_path != []:
            label, _ = SemanticKitti.open_label(self._label_path[idx])
        data = {"scan":scan, "label":label}
        if self._transform is not None:
            data = self._transform(data)
        return data

    @staticmethod
    def convert_labels(labels, reduced=True):
        if reduced:
            return np.vectorize(SemanticKitti.LEARNING_MAP.get)(labels)
        else:
            return np.vectorize(SemanticKitti.LEARNING_MAP_INV.get)(labels)
    
    @staticmethod
    def colorise(labels, img_H, img_W):
        if len(labels.shape)>1:
            proj_color_labels = np.zeros((img_H, img_W, 3), dtype=np.float)
            proj_labels = SemanticKitti.convert_labels(labels, False) # sanity check
            proj_color_labels = SemanticKitti.CMAP[labels]
            return proj_color_labels
        else:
            labels = SemanticKitti.convert_labels(labels, False) # sanity check
            color_labels = SemanticKitti.CMAP[labels]
            return color_labels
    
    @staticmethod
    def unproject_labels(proj_labels, proj_x, proj_y):
        return proj_labels[proj_y, proj_x]

    def __len__(self):
        return len(self._scan_paths)

    