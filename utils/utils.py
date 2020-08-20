import numpy as np

class utils():


    @staticmethod
    def inverse(matrix):

        new = np.zeros_like(matrix)
        new[:3,:3] = matrix[:3,:3].T
        new[:3,3] = -new[:3,:3].dot(matrix[:3,3])
        new[3,3] = 1
        
        return new
    @staticmethod
    def rigid_transform_matrix(matrix,type):
        rigid = np.zeros((4,4))
        rigid[3,3] = 1
        if type == "R|0":
            rigid[:3,:3] = matrix
        elif type == "R|t":
            rigid[:3,:4] = matrix
        else:
            
            rigid = np.hstack((matrix,np.ones((matrix.shape[0],1))))

        return rigid