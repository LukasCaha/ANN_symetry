# Artificial Neural Network

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
d = pd.read_csv('data.csv', sep=';').values
d = d[:2000,:]
d = np.array(d, dtype=np.float64)
for n in range(0,2000):
    d = np.vstack([d,[d[n,0],d[n,1],d[n,1],d[n,0],d[n,4],d[n,5],d[n,5],d[n,4],d[n,8],d[n,9],d[n,9],d[n,8],d[n,12],d[n,13],d[n,13],d[n,12],0]])
    
for r in range(0,d.shape[0]):
    d[r,16] = 1
    for p in range(0,4):
        if(d[r,0+(4*p)]!=d[r,3+(4*p)]):
            d[r,16] = 0
        if(d[r,1+(4*p)]!=d[r,2+(4*p)]):
            d[r,16] = 0
    
"""for r in range(0,d.shape[0]):
    for p in range(0,4):
        if(d[r,0+(4*p)]==d[r,3+(4*p)]):
            d[r,16] += 0.125
        if(d[r,1+(4*p)]==d[r,2+(4*p)]):
            d[r,16] += 0.125



for r in range(0,d.shape[0]):
    if(d[r,16]>0.5):
        d[r,16] = 1
    else:
        d[r,16] = 0"""

d = np.array(d, dtype=np.int64)

X = d[:, :-1]
y = d[:, 16]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))

# Adding the second hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Plotting Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plotting Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1,1])/800*100
print(accuracy,'%')

new_prediction = classifier.predict(np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))
new_prediction = (new_prediction>0.5)
print(new_prediction)



classifier.save("model.h5")
weights = classifier.get_weights()


#Save the model
# serialize model to JSON
model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")










































































