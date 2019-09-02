from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
img_rows,img_cols=28,28
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
print(x_train.shape)
input_shape=(img_rows,img_cols,1)

#We also need to make sure we operate on 32-bit floating point numbers between [0, 1], rather than unsigned integers in [0, 255]:

x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#We could do this with scikit-learn's preprocessing, but in this case it is easier to use Keras' own utility function:
from keras.utils import np_utils
n_classes=10
Y_train=np_utils.to_categorical(y_train,n_classes)
Y_test=np_utils.to_categorical(y_test,n_classes)

from keras.models import Sequential
model=Sequential()

from keras.layers import Convolution2D
n_filters=32
kernel_size=(3,3)
model.add(Convolution2D(n_filters,kernel_size[0],kernel_size[1],border_mode='valid',input_shape=input_shape))

from keras.layers import Activation
model.add(Activation('relu'))

from keras.layers import MaxPooling2D,Dropout
pool_size=(2,2)
model.add(MaxPooling2D(pool_size))
model.add(Dropout(0.25))

from keras.layers import Flatten,Dense
model.add(Flatten())
model.add(Dense(n_classes))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

#model.fit(x_train,Y_train,batch_size=128,nb_epoch=12,verbose=1,validation_data=(x_test,Y_test))

#model.evaluate(x_test,Y_test,verbose=0)
