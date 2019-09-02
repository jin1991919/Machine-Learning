from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
docs = ['Volkswagen recalls 679,000 US vehicles to fix electrical problem that could cause cars to roll away',
		'Titanium Apple Card shows signs of wear after just one month',
		'Bill Gates Says This Type of AI Will Be Worth “10 Microsofts”',
		'Apples iPhone 11 Beats Samsung Galaxy Note 10 With Two Features',
		'Amazon drops prices on Bose, Sony, Samsung, and Yamaha sound bars for Labor Day',
		]
# define class labels
labels = array([0,0,1,1,1])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 20
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 5, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
