import cv2
import os
import numpy as np

filepath = 'xixi.jpg'

mask = np.zeros((330, 380), np.uint8)

ct = np.load('xixi.npz')

#print(ct['cn2'])
cn2 = ct['cn2']
cn1 = ct['cn1']
print(cn2.shape)
cn2 = cn2.astype(np.int32)


# mask[cn2[:,0,1], cn2[:, 0,0]] = 255

cn = []
cn.append(cn2)
cn.append(cn1)

# cn2 = np.array( [[[10,10],[100,10],[100,100],[10,100]]], dtype=np.int32 )
cv2.fillPoly(mask, cn, color=255)
cv2.imwrite('mimi.jpg', mask)








