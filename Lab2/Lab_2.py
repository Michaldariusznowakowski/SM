import numpy as np
import matplotlib.pyplot as plt
import cv2

img = plt.imread('Lab2/data/A1.png')
print(img.dtype)
print(img.shape)
print(np.min(img),np.max(img))

#img2 = cv2.imread('nazwa_pliku.roz')

def imgToUInt8(img):
    return img/255.0

def imgToFloat(img):
    return (img*255).astype("uint8")



np.issubdtype(img.dtype,np.integer)
np.issubdtype(imgToUInt8(img).dtype,np.unsignedinteger)
np.issubdtype(imgToFloat(img).dtype,np.floating)

plt.imshow(img)
plt.show() 



R=img[:,:,0]
plt.imshow(R)
plt.show()

#Jakie wartości powinny przyjąć w zależności od naszych typów danych?
plt.imshow(R, cmap=plt.cm.gray, vmin=0, vmax=200)
# plt.imshow(R, cmap=plt.cm.gray, vmin=0.0, vmax=200.0)
