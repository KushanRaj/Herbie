#imports
import numpy as np

import os
#function to convert PointCloud to BEV
class BEV():


    def __init__(self,cfg,scan):
        '''
        Create class with lidar scan all operations will be performed
        set boundary values from configuration file
        '''
        
        self.IMG_SIZE = cfg['imgsize']

        #define boundaries for each dimension (according to the Complex Yolo Paper)
        self.X = cfg['X']
        self.x = cfg['x']
        self.Y = cfg['Y']
        self.y = cfg['y']
        self.Z = cfg['Z']
        self.z = cfg['z']
        
        #choose points witihin valid range
        #discretize the Point Cloud
        self.PC = self.discretize(self.cuttoff(scan))
        

    def discretize(self,PC_0):
        f'''
        transform points from continuous 2D space to discrete 2D space:
        Stretch current {self.X - self.x} size to img size:{self.IMG_SIZE} 
        ''' 
        PC = np.copy(PC_0)
        
        #discretize the X & Y dimenions as pixels
        dsc_x = (self.X -self.x)/(self.IMG_SIZE -1)
        dsc_y = (self.Y - self.y)/(self.IMG_SIZE -1)
        
        PC[:,0] = np.floor(((PC[:,0] - self.x)/dsc_x)).astype(np.int32)
        PC[:,1] = np.floor(((PC[:,1] - self.y)/dsc_y)).astype(np.int32)
        return PC
    
    def cuttoff(self,PC_0):
        '''
        Choose only valid points according to defined boundary range
        Store the new maximum and minimum value of the PointCloud
        '''
        PC_0 = PC_0[np.where((PC_0[:,2]<=self.Z ) 
                            &(PC_0[:,2]>=self.z) 
                            &(PC_0[:,0]>=self.x) 
                            &(PC_0[:,0]<=self.X) 
                            &(PC_0[:,1]>=self.y) 
                            &(PC_0[:,1]<=self.Y))
                ]
        self.X = np.max(PC_0[:,0])
        self.Y = np.max(PC_0[:,1])
        self.y = np.min(PC_0[:,1])
        self.x = np.min(PC_0[:,0])
        self.z = np.min(PC_0[:,2])
        self.Z = np.max(PC_0[:,2])

        return PC_0
    def check_cuttoff(self,pts):
        '''
        Check if the points are within the valid range
        '''
        return ((pts[:,0]>=self.x)
               &(pts[:,0]<=self.X)
               &(pts[:,1]>=self.y)
               &(pts[:,1]<=self.Y))

    def __call__(self):
        '''
        get bird eyes view of the lidar scan
        in -> 4D scan
        out -> 2D Bird's Eye View RGB image
        '''
        
        #get the scan
        PC = self.PC.astype(np.float32)
        
        #Normalize heights and reflectance
        PC[:,2] = ((PC[:,2] - (self.z))/((self.Z-self.z)))
        PC[:,3] /= np.max(PC[:,3])
        
        #create image of desired size with 3 channels
        BEV = np.zeros(shape=(self.IMG_SIZE,self.IMG_SIZE,3))

        #sort according to height
        PC = PC[np.argsort(-PC[:,2])]    
        
        #find qunique points in discrete 2D space with max height
        #store the counts
        _,index,count = np.unique(PC[:,0:2],axis=0,return_index=True,return_counts=True)
        
        #find the normalize density
        count = np.minimum(1,np.log(count+1)/64)
        
        #Density Map
        BEV[PC[index][:,0].astype(np.int),PC[index][:,1].astype(np.int),0] = count

        #Height Map
        BEV[PC[index][:,0].astype(np.int),PC[index][:,1].astype(np.int),2]= PC[index][:,2]

        #sort according to reflectance
        PC = PC[np.argsort(-PC[:,3])]

        #find qunique points in discrete 2D space with max reflectance
        _,index= np.unique(PC[:,0:2],axis=0,return_index=True)

        #Intensity Map
        BEV[PC[index][:,0].astype(np.int),PC[index][:,1].astype(np.int),1] = PC[index][:,3]


        '''
        for i,c in zip(unique.astype(np.int32),count):
            #find all points in the PointCloud which are transformed to the same pixel
            index = np.where((PC[:,0] == i[0]) & (PC[:,1] == i[1]))
            
            #Height Map
            BEV[i[0],i[1],2] = np.max(PC[index][:,2]) #Choose the point with the heighest height
            #Intensity Map
            BEV[i[0],i[1],1] = np.max(PC[index][:,3]) #Choose the point with the heighest intensity
            #Density Map
            BEV[i[0],i[1],0] = c #Calculate the normalized density of points at the pixel
        
        '''
        return BEV

    def get_2Dbox(self,points,yaw):
        
        pts = points[:4,:2]
        
        
        boundary = self.check_cuttoff(pts)
        if boundary.all():
            pts =  self.discretize(np.hstack((np.array(pts).reshape((4,2)),np.ones((4,2))))).astype(np.int32)[:,:2]
            
            
            g = pts[np.argsort(pts[:,0])]
            #adj = g[1:][np.argsort([(i - g[0])[1]/(i - g[0])[0] for i in g[1:]])]
            #h = np.sqrt((adj[-1]-g[0])[0]**2 + (adj[-1]-g[0])[1]**2)
            #w = np.sqrt((adj[0]-g[0])[0]**2 + (adj[0]-g[0])[1]**2)
            
            c = np.cos(yaw-np.pi/2)
            s = np.sin(yaw-np.pi/2)
            
            pts = (pts-np.mean(pts,0)).dot(np.array([[c,-s],[s,c]]).T) + np.mean(pts,0)
            
            f = pts[np.argsort(pts[:,0])]
            p = g[1:][np.argsort([(i - g[0])[1]/(i - g[0])[0] for i in f[1:]])]
            dim = p[1]-f[0]
            
            
                
            return np.hstack((np.mean(pts,0),dim)),True
        else:
            return pts,False

        
    

