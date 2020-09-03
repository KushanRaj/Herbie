import cv2
import numpy as np



def draw_bev(image,pts,bev_with_box):

        flip = np.array([[0,1],[1,0]])
        #if bev_with_box:
        
        for pt in pts:
                pt = get_corner(pt[-4:-2],pt[-2],pt[-1],pt[-5])
                image = cv2.polylines(image,[pt.dot(flip.T).astype(np.int32)],True,(255,255,255))
                image = cv2.circle(image,tuple(flip.dot(np.mean(pt,0).T).astype(np.int32)),2,(255,255,255))

        cv2.imshow('Birds-Eye-View',image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

def get_corner(point,w,h,yaw):
    
    
    points = np.array([[-w,h] , [w,h], [w,-h], [-w,-h]])/2
    
    points = points.reshape((4,2))
    c = np.cos(yaw)
    s = np.sin(yaw)

    points = point + points.dot(np.array([[c,-s],[s,c]]).T)
    
    return(points)



