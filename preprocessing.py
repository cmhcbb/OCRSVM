from mnist import load_mnist
import numpy as np
import cv2
#img=cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
#print 'ori'
#print img
def deskew(img):
    (h, w) = img.shape[:2]
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*w*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(w, h),flags=affine_flags)
    return img

#deskewed_image = deskew(img)
#print 'deskew'
#print deskewed_image
#[feature,lable]=load_mnist("training")
#[tfeature,tlable]=load_mnist("testing")
#print feature[0]