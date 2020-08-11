from torch.utils.data import Dataset
import numpy as np
import os 
from glob import glob
from transformation import spherical_proj

LABELS = { 
  0 : "unlabeled",
  1 : "outlier",
  10: "car",
  11: "bicycle",
  13: "bus",
  15: "motorcycle",
  16: "on-rails",
  18: "truck",
  20: "other-vehicle",
  30: "person",
  31: "bicyclist",
  32: "motorcyclist",
  40: "road",
  44: "parking",
  48: "sidewalk",
  49: "other-ground",
  50: "building",
  51: "fence",
  52: "other-structure",
  60: "lane-marking",
  70: "vegetation",
  71: "trunk",
  72: "terrain",
  80: "pole",
  81: "traffic-sign",
  99: "other-object",
  252: "moving-car",
  253: "moving-bicyclist",
  254: "moving-person",
  255: "moving-motorcyclist",
  256: "moving-on-rails",
  257: "moving-bus",
  258: "moving-truck",
  259: "moving-other-vehicle"
}

COLOR_MAP = {
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]
}

# ratio of points in a class with respect to the total number
CONTENT = {
  0: 0.018889854628292943,
  1: 0.0002937197336781505,
  10: 0.040818519255974316,
  11: 0.00016609538710764618,
  13: 2.7879693665067774e-05,
  15: 0.00039838616015114444,
  16: 0.0,
  18: 0.0020633612104619787,
  20: 0.0016218197275284021,
  30: 0.00017698551338515307,
  31: 1.1065903904919655e-08,
  32: 5.532951952459828e-09,
  40: 0.1987493871255525,
  44: 0.014717169549888214,
  48: 0.14392298360372,
  49: 0.0039048553037472045,
  50: 0.1326861944777486,
  51: 0.0723592229456223,
  52: 0.002395131480328884,
  60: 4.7084144280367186e-05,
  70: 0.26681502148037506,
  71: 0.006035012012626033,
  72: 0.07814222006271769,
  80: 0.002855498193863172,
  81: 0.0006155958086189918,
  99: 0.009923127583046915,
  252: 0.001789309418528068,
  253: 0.00012709999297008662,
  254: 0.00016059776092534436,
  255: 3.745553104802113e-05,
  256: 0.0,
  257: 0.00011351574470342043,
  258: 0.00010157861367183268,
  259: 4.3840131989471124e-05
}

# objects which are not identifiable from a single scan are mapped to their closest
LEARNING_MAP = {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5     # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

# invert above feature map
LEARNING_MAP_INV = {
  0: 0,      # "unlabeled", and others ignored
  1: 10,     # "car"
  2: 11,     # "bicycle"
  3: 15,     # "motorcycle"
  4: 18,     # "truck"
  5: 20,     # "other-vehicle"
  6: 30,     # "person"
  7: 31,     # "bicyclist"
  8: 32,     # "motorcyclist"
  9: 40,     # "road"
  10: 44,    # "parking"
  11: 48,    # "sidewalk"
  12: 49,    # "other-ground"
  13: 50,    # "building"
  14: 51,    # "fence"
  15: 70,    # "vegetation"
  16: 71,    # "trunk"
  17: 72,    # "terrain"
  18: 80,    # "pole"
  19: 81     # "traffic-sign"
}

# classes to ignore
LEARNING_IGNORE = { 
  0: True,      # "unlabeled", and others ignored
  1: False,     # "car"
  2: False,     # "bicycle"
  3: False,     # "motorcycle"
  4: False,     # "truck"
  5: False,     # "other-vehicle"
  6: False,     # "person"
  7: False,     # "bicyclist"
  8: False,     # "motorcyclist"
  9: False,     # "road"
  10: False,    # "parking"
  11: False,    # "sidewalk"
  12: False,    # "other-ground"
  13: False,    # "building"
  14: False,    # "fence"
  15: False,    # "vegetation"
  16: False,    # "trunk"
  17: False,    # "terrain"
  18: False,    # "pole"
  19: False     # "traffic-sign"
}

# sequences in split types
SPLIT = { 
  "train" : [0,1,2,3,4,5,6,7,9,10],
  "val" : [8],
  "test"  : [11,12,13,14,15,16,17,18,19,20,21]
}

# sensor configuration
SENSOR_CONFIG = {
    "name": "HDL64",
    "type": "spherical",
    "fov_up": 3,
    "fov_down": -25
}

# projected image properties
IMG_PROP = {
    # range, x, y, z signal
    "img_means": [12.12, 10.88, 0.23, -1.04, 0.21],
    "img_stds": [12.32, 11,47, 6,91, 0.86, 0.16] 
}

AVAILABLE_TRANSFORMS = {
    "spherical_proj": spherical_proj.SphericalProjection
}

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

    