import cv2
import numpy as np

class visualize():

    def draw_bev(self,image,pts,bev_with_box):

        
        if bev_with_box:
            for pt in pts:
                image = cv2.polylines(image,[pt],True,(255,255,255))

        #cv2.circle(image2,(90,255),5,(255,255,255))
        cv2.imshow('Birds-Eye-View',image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()