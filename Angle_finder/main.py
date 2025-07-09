from Models import Angle_CNN_2D
import cv2
import numpy as np
#fedora=cv2.imread(r"C:\Code\KD_PRACT\KD_Summer_Work\Angle_finder\test_fedora.png",cv2.IMREAD_UNCHANGED)#сохраняет в исходном виде

img=np.ones((512,512,3),np.uint8)


#рисуем линии


cv2.line(img,(0,100),(512, 100), (0, 0, 255), 5)#вверх
cv2.line(img,(0,180),(512, 100), (0, 0, 255), 5)#вверх
cv2.line(img,(0,100),(512, 100), (0, 0, 255), 5)#вверх
cv2.line(img,(0,100),(512, 100), (0, 0, 255), 5)#вверх


cv2.imshow('dst',img)
cv2.waitKey(0)
