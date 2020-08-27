import cv2
import numpy as np

class visualize():

    def draw_bev(self,image,pts,bev_with_box):

        flip = np.array([[0,1],[1,0]])
        if bev_with_box:
            for pt in pts:
                
                image = cv2.polylines(image,[pt.reshape(4,2).dot(flip.T).astype(np.int32)],True,(255,255,255))

        #cv2.circle(image2,(90,255),5,(255,255,255))
        cv2.imshow('Birds-Eye-View',image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()