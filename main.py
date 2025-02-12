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