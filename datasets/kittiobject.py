from torch.utils.data import Dataset
import numpy as np
import os 
from glob import glob
from utils import utils,common

import torch
from transformation.BEV import BEV
import torch


class KittiObject(Dataset):

    
    def __init__(self, config, split="train"):
        super().__init__()
        self.root = config["root"]
        self.IMG_SIZE = config["imgsize"]
        self.max = config["max_label"]
        self.config = config
        self.num_classes = config["num_classes"]
        

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
        
        assert len(self._scan_paths) == len(self._label_path)
                 
  
    def read_file(self,idx):

        label = np.array([object_.split(" ") for object_ in [objects.rstrip() for objects in open(self._label_path[idx])]])
        label[:,1:] = label[:,1:].astype(np.float32)
        label = label[np.where(label[:,0]!= 'DontCare')]

        self.clid = [self.config[i] for i in label[:,0]]
        

        self.lidar = np.fromfile(self._scan_paths[idx],dtype=np.float32).reshape((-1,4))
         
        self.cam_pos = label[:,11:14].astype(np.float32)
        self.box_dim = label[:,8:11].astype(np.float32)
        self.Box2D = label[:,4:8].astype(np.float32)
        self.yaw = label[:,-1].astype(np.float32)

        Matrices = [matrix.split(" ") for matrix in open(os.path.join(self._calib_path[idx]))]
        self.P2L = np.array(Matrices[2])[1:].reshape((3,4)).astype(np.float32)
        self.P2R = np.array(Matrices[3])[1:].reshape((3,4)).astype(np.float32)
        self.R = utils.rigid_transform_matrix(np.array(Matrices[4])[1:].reshape((3,3)).astype(np.float32),"R|0")
        self.T = utils.rigid_transform_matrix(np.array(Matrices[5])[1:].reshape((3,4)).astype(np.float32),"R|t")
        self.T_inv = utils.inverse(self.T)
        self.R_inv = utils.inverse(self.R) 

    
    
    
    


    def towardsim(self,vectors,pos,lim):

        if pos != lim:
            if pos == 'ref':
                pos = 'rect'
                
                vectors = self.towardsim(vectors.dot(self.R.T),pos,lim)
            if pos == 'velo':
                pos = 'ref'
                
                vectors =  self.towardsim(vectors.dot(self.T.T),pos,lim)
            if pos == 'rect':
                pos = 'im'
                vectors = self.towardsim(vectors.dot(self.P2L.T),pos,lim)

        
        return vectors
    
    def towardsvelo(self,vectors,pos,lim):

        if pos != lim:
             
             if pos == 'ref':
                pos = 'velo'
                 
                vectors = self.towardsvelo(vectors.dot(self.T_inv.T),pos,lim)
             if pos == 'rect':
                pos = 'ref'
                 
                vectors = self.towardsvelo(vectors.dot(self.R_inv.T),pos,lim)
            
        return vectors
        
    
    def get_BEV_box(self,obj_id,bev):


        clid = [0 for i in range(self.num_classes)]
        clid[self.clid[obj_id]] = 1

        x = np.array([self.box_dim[obj_id,2],self.box_dim[obj_id,2],-self.box_dim[obj_id,2],-self.box_dim[obj_id,2],-self.box_dim[obj_id,2],
                      -self.box_dim[obj_id,2],self.box_dim[obj_id,2],self.box_dim[obj_id,2]])/2


        y = np.array([0,0,0,0,-self.box_dim[obj_id,0],-self.box_dim[obj_id,0],self.box_dim[obj_id,0],self.box_dim[obj_id,0]])


        z = np.array([-self.box_dim[obj_id,1],self.box_dim[obj_id,1],self.box_dim[obj_id,1],-self.box_dim[obj_id,1],-self.box_dim[obj_id,1],
                      self.box_dim[obj_id,1],self.box_dim[obj_id,1],-self.box_dim[obj_id,1]])/2

        corners = np.vstack((x,y,z))

        s = np.sin(self.yaw[obj_id])
        c = np.cos(self.yaw[obj_id])
            
        rotation = np.matrix([[c,0,s],[0,1,0],[-s,0,c]])
        translation = self.cam_pos[obj_id]
            
        matrix = np.zeros((4,4))
        matrix[:3,:3] = rotation
        matrix[:3,3] = translation
        matrix[3,3] = 1
        boxes,boundary = bev.get_2Dbox(self.towardsvelo(utils.rigid_transform_matrix(corners.T,None).dot(matrix.T),'rect','velo'),self.yaw[obj_id])

        return np.concatenate((clid,[np.pi/2-self.yaw[obj_id]]
                                            ,boxes.flatten()
                                            )),boundary
    
       
    def get_BEV_boxes(self,bev):
        boxes = []
        for i,_ in enumerate(self.cam_pos):
            box,boundary = self.get_BEV_box(i,bev)
            if boundary:
                boxes.append(box)


        return torch.tensor(boxes).float()

    def __getitem__(self, idx):

        self.read_file(idx)
        
        bev = BEV(self.config,self.lidar)
        scan = bev()
        label = torch.zeros((self.max,12))
        
        boxes = self.get_BEV_boxes(bev)
        if boxes.size(0):
            
            label[:boxes.size(0)] = boxes
            '''
            if self._label_path != []:
                label = self.get_2D_label(self._label_path[idx],self._calib_path[idx],self.config)
            '''
            data = {"scan":scan, "target":label,"n_box":boxes.size(0)}
            
            return data

    def __len__(self):
        return len(self._scan_paths)