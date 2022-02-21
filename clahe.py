import cv2
import numpy as np

def enhance_outdated(image : np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    out = np.zeros(image.shape)
    for i in range(3):

        # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        out[:, :, i] = clahe.apply((image[:, :, i]))
    return out

class CLAHE():
    def __init__(self):
        pass 
    
    def __call__(self, image: np.ndarray, bboxes = None, labels = None):
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
        out = np.zeros(image.shape)
        for i in range(3):

            # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
            out[:, :, i] = clahe.apply((image[:, :, i]))
        return out, bboxes, labels
