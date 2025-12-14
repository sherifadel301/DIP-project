import numpy as np
import cv2

def extract_color(self,image):
  feature=[]
  image=cv2.resize(image,(128,128))
  if image.shape[-1]==3 :
    for i in range(3):
      hist=cv2.calcHist([image],[i],None,[256],[0,256])
      feature.extend(hist.flatten())
  else:
    hist=cv2.calcHist([image],[0],None,[256],[0,256])
    feature=hist
  return np.array(feature.flatten())


def extract_edge(self,image):
  image=cv2.resize(image,(64,64))
  image=cv2.Canny(image,50,200)
  return np.array(image.flatten())
