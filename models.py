#models
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#tools
from sklearn.metrics import accuracy_score
import numpy as np
from google.colab.patches import cv2_imshow
import cv2
import os
import re


class Image_processing_models:
  __scaler = StandardScaler()
  __output={"human":0,"animal":1,"plant":2}
  
  def __init__(self,folder_animal_path,folder_human_path,folder_planet_path,model="KNN"):
    self.x_train=[]#data as feature
    self.y_train=[]#type
    self.input_file=None
    self.model=model
    folder_human=os.listdir(folder_human_path)
    folder_animal=os.listdir(folder_animal_path)
    folder_planet=os.listdir(folder_planet_path)

    for i in folder_human:
      image=cv2.imread(f'{data_set[0]}/{i}')
      self.main_process(image)
      self.y_train.append(self.__output['human'])

    for i in folder_animal:
      image=cv2.imread(f'{data_set[1]}/{i}')
      self.main_process(image)
      self.y_train.append(self.__output['animal'])
    
    for i in folder_planet:
      image=cv2.imread(f'{data_set[2]}/{i}')
      self.main_process(image)
      self.y_train.append(self.__output['plant'])


  def extract_color(self,image):
    feature=[]
    if image.shape[-1]==3 :
      hist=self.__get_histogram(image)
      for i in range(3):
        feature.extend(hist[i].flatten())
    else:
      feature.append(self.__get_histogram(image).flatten())
    
    return np.array(feature)


  def draw_histogram(self,image=None):
    if image is None:
      image=cv2.imread(self.input_file)

    hist=self.__get_histogram(image)
    for i in hist:
      plt.plot(i)
    plt.show()

  def __get_histogram(self,image):
    image=cv2.resize(image,(128,128))
    if image.shape[-1]==3 :
      histogram_channels=[]
      for i in range(3):
        hist=cv2.calcHist([image],[i],None,[256],[0,256])
        histogram_channels.append(hist)
      return histogram_channels
    else:
      hist=cv2.calcHist([image],[0],None,[256],[0,256])
      return hist


  def extract_edge(self,image):
    image=cv2.resize(image,(64,64))
    image=cv2.Canny(image,50,200)
    return np.array(image.flatten())

  def main_process(self,image):
      image=cv2.resize(image,(64,64))
      gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
      fetau=np.concatenate([self.extract_edge(gray_image),self.extract_color(image)])
      self.x_train.append(fetau)

  def fit_model(self):
    #normaliz data
    self.x_train = self.__scaler.fit_transform(self.x_train)

    if(self.model=='KNN'):
      self.model=KNeighborsClassifier(n_neighbors=3)
      self.model.fit(self.x_train,self.y_train)
    elif (self.model=='KMeans'):
      self.model=KMeans(3)
      self.model.fit_predict(self.x_train)


  def test_model(self,path_image):
    self.input_file=path_image
    image=cv2.imread(self.input_file)
    image=cv2.resize(image,(125,125))
    gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    fetau=np.concatenate([self.extract_edge(gray_image),self.extract_color(image)])
    
    number_output=self.model.predict([fetau])
    type_output=None
    if(number_output==0):
      type_output="human"
    elif(number_output==1):
      type_output="animal"
    elif(number_output==2):
      type_output="plant"
    return number_output,type_output

  def test_accuorce(self,paths,y_test):
    y_pred=[]
    for path_image in paths:
      number_object,type_object=self.test_model(path_image)
      y_pred.append(number_object)
    return accuracy_score(y_test,y_pred)


data_set=[
    "/content/human_train",
    "/content/animal_train",
    "/content/planet_train",
]

model=Image_processing_models(data_set[1],data_set[0],data_set[2],'KNN')
model.fit_model()
