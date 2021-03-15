import numpy as np
from scipy.io import loadmat
import cv2
import math

class camera_model:
    """
    class implementing a pinhole camera model  with parameters loaded from a matlab file. It has functions to project a point on an image, rectify an image ...
    """

    def __init__(self,matfile):
        self.matInfo = loadmat(matfile)
        self.f = self.matInfo['fc'].flatten()
        self.c = self.matInfo['cc'].flatten()
        self.imgSize = np.array([self.matInfo['nx'][0,0],self.matInfo['ny'][0,0]])

        self.K = np.array([ [self.f[0],0,self.c[0]] , [0,self.f[1],self.c[1]] , [0,0,1] ])
        self.kc = self.matInfo['kc'].flatten()

    def getImSize(self):
        return self.imgSize[0], self.imgSize[1]

    def getF(self):
        return self.f

    def getC(self):
        return self.c
