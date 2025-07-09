from Models import Angle_CNN_2D
import cv2
import numpy as np
#fedora=cv2.imread(r"C:\Code\KD_PRACT\KD_Summer_Work\Angle_finder\test_fedora.png",cv2.IMREAD_UNCHANGED)#сохраняет в исходном виде

shape=(1100,762,3)
img=np.ones(shape,np.uint8)
weigj=shape[1]
hight=shape[0]

up=100
left=100
right=weigj-100
down=hight-180
print(up,left,right,down)


#рисуем линии



cv2.line(img,(0,up),(weigj, up), (0, 0, 255), 5)#вверх
cv2.line(img,(left,0),(left, hight), (0, 0, 255), 5)#лево
cv2.line(img,(right,0),(right, hight), (0, 0, 255), 5)#право
cv2.line(img,(0,down),(weigj, down), (0, 0, 255), 5)#низ

cv2.imwrite('output.jpg', img)  


cv2.imshow('dst',img)
cv2.waitKey(0)
