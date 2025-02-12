import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Input
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
print(xtrain.shape)
plt.imshow(xtrain[0])
plt.savefig('./pngFiles/data_1.png')
'''Conventional Neural Network we need to flatten the images into one-dimensional vectors '''
num_of_pixels = xtrain.shape[1]*xtrain.shape[2]
xtrain = xtrain.reshape(xtrain.shape[0],num_of_pixels).astype('float32')
xtest = xtest.reshape(xtest.shape[0],num_of_pixels).astype('float32')
print(xtrain.shape)
print(xtest.shape)
xtrain=xtrain/255
xtest=xtest/255
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
num_of_classes = ytest.shape[1]
print(num_of_classes)
print(num_of_pixels)
def classification_model():
    model=Sequential()
    model.add(input(shape=num_of_pixels))