from torch.utils.data import Dataset
import numpy as np
import os 
from glob import glob
from transformation.Frame import Frame
import torch
from transformation.BEV import BEV

class BEVDetection(Dataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.root = config["root"]
        self.config = config
        self._normalise = config["normalise"]
        

        if split not in list(config["SPLIT"]):
            raise ValueError(f"split {split} does not exist. choose from train or test")
        else:
            self._load_paths(split)
        

    def _load_paths(self, split):

        
        self._scan_paths = []
        self._label_path = []
        self._calib_path = []
        self._image_path = []
        
        self._scan_paths.extend(sorted(glob(os.path.join(self.root,split,"velodyne", '*.bin'))))
        self._label_path.extend(sorted(glob(os.path.join(self.root,split,"label_2", '*.txt'))))
        self._calib_path.extend(sorted(glob(os.path.join(self.root,split,"calib", '*.txt'))))
        self._image_path.extend(sorted(glob(os.path.join(self.root,split,"image_2", '*.png'))))
        
        if len(self._scan_paths) != len(self._label_path):
            raise ValueError((f"number of scans {len(self._scan_paths)} not equal to number of labels {len(self._label_path)}"))     
  
    
    @staticmethod
    def get_bevfrom_scan(file_name,config):
        """
        open scan file - [x, y, z, remissions]
        returns point coordinates, remissions
        """
        scan = np.fromfile(file_name, dtype=np.float32)
        scan = scan.reshape((-1, 4)) # just for the sake of it
        
        return BEV(config).transform_to_BEV(scan)
    
    @staticmethod
    def get_2D_label(lfile,cfile,config):
        """
        open label file - [semantic label(first 16 bits), instance label(last 16 bits)]
        returns semantic label, instance label
        """
        
        label,points = Frame(lfile,cfile,config)
        return label,points

    def __getitem__(self, idx):
        scan = self.get_bevfrom_scan(self._scan_paths[idx],self.config)
        label = None
        if self._label_path != []:
            label,points = self.get_2D_label(self._label_path[idx],self._calib_path[idx],self.config)
        data = {"scan":scan, "label":[label,points]}
        
        return data

    def __len__(self):
        return len(self._scan_paths)