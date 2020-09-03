#imports
import numpy as np

import os
#function to convert PointCloud to BEV
class BEV():


    def __init__(self,cfg):
        
        self.IMG_SIZE = cfg['imgsize']

        #define boundaries for each dimension (according to the Complex Yolo Paper)
        self.X = cfg['X']
        self.x = cfg['x']
        self.Y = cfg['Y']
        self.y = cfg['y']
        self.Z = cfg['Z']
        self.z = cfg['z']

    def discretize(self,PC_0):

        PC = np.copy(PC_0)

        #discretize the X & Y dimenions as pixels
        dsc_x = (self.X - self.x)/(self.IMG_SIZE -1)
        dsc_y = (self.Y - self.y)/(self.IMG_SIZE -1)
        
        PC[:,0] = np.floor(((PC[:,0] - self.x)/dsc_x)).astype(np.int32)
        PC[:,1] = np.floor(((PC[:,1] - self.y)/dsc_y)).astype(np.int32)
        return PC
  
    def transform_to_BEV(self,PC):
        
        #Select only points within the boundary
        PC = PC[np.where((PC[:,2]<=self.Z )
                          &(PC[:,2]>=self.z)
                          &(PC[:,0]>=self.x)
                          &(PC[:,0]<=self.X)
                          &(PC[:,1]>=self.y)
                          &(PC[:,1]<=self.Y))
                ]
                            
        PC = self.discretize(PC)
        
        #Normalize heights and reflectance
        PC[:,2] = (PC[:,2] - (self.z))/((self.Z-self.z))
        PC[:,3] /= np.max(PC[:,3])
    
        #find density and unique pixels
        unique,count = np.unique(PC[:,0:2],axis=0,return_counts=True)
        #normalize density
        count = np.minimum(1,np.log(count+1)/64)

        #create image of desired size with 3 channels
        BEV = np.zeros(shape=(self.IMG_SIZE,self.IMG_SIZE,3))
        
        for i,c in zip(unique.astype(np.int32),count):
            #find all points in the PointCloud which are transformed to the same pixel
            index = np.where((PC[:,0] == i[0]) & (PC[:,1] == i[1]))
        
            #Height Map
            BEV[i[0],i[1],2] = np.max(PC[index][:,2]) #Choose the point with the heighest height
            #Intensity Map
            BEV[i[0],i[1],1] = np.max(PC[index][:,3]) #Choose the point with the heighest intensity
            #Density Map
            BEV[i[0],i[1],0] = c #Calculate the normalized density of points at the pixel
        

        return BEV

    def get_2Dbox(self,points,yaw):
        
        pts = points[:4,:2]
        


        pts =  self.discretize(np.hstack((np.array(pts).reshape((4,2)),np.ones((4,2))))).astype(np.int32)[:,:2]
        
        
        g = pts[np.argsort(pts[:,0])]
        adj = g[1:][np.argsort([(i - g[0])[1]/(i - g[0])[0] for i in g[1:]])]
        #h = np.sqrt((adj[-1]-g[0])[0]**2 + (adj[-1]-g[0])[1]**2)
        #w = np.sqrt((adj[0]-g[0])[0]**2 + (adj[0]-g[0])[1]**2)
        
        c = np.cos(yaw-np.pi/2)
        s = np.sin(yaw-np.pi/2)
        
        pts = (pts-np.mean(pts,0)).dot(np.array([[c,-s],[s,c]]).T) + np.mean(pts,0)
        
        f = pts[np.argsort(pts[:,0])]
        p = g[1:][np.argsort([(i - g[0])[1]/(i - g[0])[0] for i in f[1:]])]
        dim = p[1]-f[0]
        
        
            
        return np.hstack((np.mean(pts,0),dim))

        
    

