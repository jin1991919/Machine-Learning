from keras.models import Sequential
#The simplest type of model is the Sequential model, which arranges the different layers of the neural net in a linear stack, just like we did for the MLP in OpenCV:
model=Sequential()

#implemented using a Dense layer that has two inputs and one output.
from keras.layers import Dense
model.add(Dense(1,activation='tanh',input_dim=2,kernel_initializer='zeros'))

#we will choose stochastic gradient descent as an optimizer, the mean squared error as a cost function, and accuracy as a scoring function:
model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

from sklearn.datasets.samples_generator import make_blobs
x,y=make_blobs(n_samples=100,centers=2,cluster_std=2.2,random_state=42)

#Here, we can also choose how many iterations to train for (epochs), how many samples to present before we calculate the error gradient (batch_size), whether to shuffle the dataset (shuffle), and whether to output progress updates (verbose):
model.fit(x,y,epochs=400,batch_size=100,shuffle=False,verbose=0)

model.evaluate(x,y)


