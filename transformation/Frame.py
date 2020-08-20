import os
import numpy as np
from utils.utils import utils #needs new naming
from transformation.BEV import BEV
from visualisation.visualize import visualize

class Frame():
    
    @staticmethod
    def towardsim(vectors,pos,lim,c_class):

        if pos != lim:
            if pos == 'ref':
                pos = 'rect'
                
                vectors = Frame.towardsim(vectors.dot(c_class.R.T),pos,lim,c_class)
            if pos == 'velo':
                pos = 'ref'
                
                vectors =  Frame.towardsim(vectors.dot(c_class.T.T),pos,lim,c_class)
            if pos == 'rect':
                pos = 'im'
                vectors = Frame.towardsim(vectors.dot(c_class.P.T),pos,lim,c_class)

        
        return vectors
    
    @staticmethod
    def towardsvelo(vectors,pos,lim,c_class):

        if pos != lim:
             
             if pos == 'ref':
                pos = 'velo'
                 
                vectors = Frame.towardsvelo(vectors.dot(c_class.T_inv.T),pos,lim,c_class)
             if pos == 'rect':
                pos = 'ref'
                 
                vectors = Frame.towardsvelo(vectors.dot(c_class.R_inv.T),pos,lim,c_class)
            
        return vectors
    
    @staticmethod
    def get_BEV_box(obj_id,l_class,c_class,config):


        flip = np.array([[0,1],[1,0]])

        x = np.array([l_class.box_dim[obj_id,2],l_class.box_dim[obj_id,2],-l_class.box_dim[obj_id,2],-l_class.box_dim[obj_id,2],-l_class.box_dim[obj_id,2],
                      -l_class.box_dim[obj_id,2],l_class.box_dim[obj_id,2],l_class.box_dim[obj_id,2]])/2


        y = np.array([0,0,0,0,-l_class.box_dim[obj_id,0],-l_class.box_dim[obj_id,0],l_class.box_dim[obj_id,0],l_class.box_dim[obj_id,0]])


        z = np.array([-l_class.box_dim[obj_id,1],l_class.box_dim[obj_id,1],l_class.box_dim[obj_id,1],-l_class.box_dim[obj_id,1],-l_class.box_dim[obj_id,1],
                      l_class.box_dim[obj_id,1],l_class.box_dim[obj_id,1],-l_class.box_dim[obj_id,1]])/2

        corners = np.vstack((x,y,z))

        s = np.sin(l_class.yaw[obj_id])
        c = np.cos(l_class.yaw[obj_id])
            
        rotation = np.matrix([[c,0,s],[0,1,0],[-s,0,c]])
        translation = l_class.cam_pos[obj_id]
            
        matrix = np.zeros((4,4))
        matrix[:3,:3] = rotation
        matrix[:3,3] = translation
        matrix[3,3] = 1
            
        return BEV(config).get_2Dbox(Frame.towardsvelo(utils.rigid_transform_matrix(corners.T,None).dot(matrix.T),'rect','velo',c_class)).dot(flip.T)
    
    @staticmethod    
    def get_BEV_boxes(l_class,c_class,config):

        return [Frame.get_BEV_box(i,l_class,c_class,config) for i,val in enumerate(l_class.cam_pos)]


    def __call__(self,label_file,calib_file,config):        

        return (Label(label_file,config).clid, self.get_BEV_boxes(Label(label_file,config),Calib(calib_file),config))


class Label():

    def init(self,label_file,config):
        
        label = np.array([object_.split(" ") for object_ in [objects.rstrip() for objects in open(label_file)]])
        label[:,1:] = label[:,1:].astype(np.float32)
        label = label[np.where(label[:,0]!= 'DontCare')]

        self.clid = config[label[0]]
          
        self.cam_pos = label[:,11:14].astype(np.float32)
        self.box_dim = label[:,8:11].astype(np.float32)
        self.Box2D = label[:,4:8].astype(np.float32)
        self.yaw = label[:,-1].astype(np.float32)   

class Calib():


    def init(self,calib_file):

        Matrices = [matrix.split(" ") for matrix in open(os.path.join(calib_file))]
        self.P2L = np.array(Matrices[2])[1:].reshape((3,4)).astype(np.float32)
        self.P2R = np.array(Matrices[3])[1:].reshape((3,4)).astype(np.float32)
        self.R = utils.rigid_transform_matrix(np.array(Matrices[4])[1:].reshape((3,3)).astype(np.float32),"R|0")
        self.T = utils.rigid_transform_matrix(np.array(Matrices[5])[1:].reshape((3,4)).astype(np.float32),"R|t")
        self.T_inv = utils.inverse(self.T)
        self.R_inv = utils.inverse(self.R) 



