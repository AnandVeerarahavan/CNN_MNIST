
# Dataset - MNIST


# Import the libraries

import tensorflow
import numpy as np 
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D , MaxPool2D , Flatten , Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Loading the dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("================================================")
print()
print("Shape of training data", x_train.shape)
print()
print("================================================")

print("================================================")
print()
print("Shape of testing data", x_test.shape)
print("================================================")
print()

# Visualizing the images
print("================================================")
print()
print("Visualizing the images")
print("================================================")
print()
plt1 = plt.imshow(x_train[5],cmap='gray')
plt.title(y_train[5])
plt.show()
print("================================================")
print()
print("Image output - ", y_train[5])
print("================================================")
print()


# Normalising the data
X_train, X_test = x_train/255 , x_test/255

# Reshaping the data to 4D
x_train = X_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000, 28, 28,1)

# STEPS:
# 1. Architecture
# 2. Compilation
# 3. Fit


# Step 1 - Architecture

# Building the CNN model

model = Sequential()

# Convolution Layer 1
model.add(Conv2D(36, 3, activation='relu', kernel_initializer='he_uniform',))
model.add(MaxPool2D())

# Convolution Layer 2
model.add(Conv2D(72, 3, activation='relu', kernel_initializer='he_uniform',))
model.add(MaxPool2D())

# Convolution Layer 3
model.add(Conv2D(144, 3, activation='relu', kernel_initializer='he_uniform',))
model.add(MaxPool2D())

# Flattening the matrix
model.add(Flatten())

# Building the ANN model

# Hidden Layer 1
model.add(Dense(128, activation='relu',kernel_initializer='he_uniform'))

# Hidden Layer 2
model.add(Dense(64, activation='sigmoid'))

# Hidden Layer 3
model.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))

# Output Layer(ANN)
model.add(Dense(10, activation='softmax'))


# Step 2 - Model Compilation
model.compile(optimizer='adam', loss=tensorflow.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# Step -3 Fitting the model
model.fit(x_train,y_train, epochs=10, batch_size=32)

# Evaluating the model on test data
print("==================================================================================")
print()
print("Model evaluating...")
model.evaluate(x_test, y_test)
print("==================================================================================")
print()

# Prediction variable y_pred
y_pred = model.predict(x_test)

# Predicting the 100 image in the test data
print("==================================================================================")
print()
print("Prediction : 1 - ", y_pred[100])
print()
print("==================================================================================")
# Double checking the test image for users to visualize the correct prediction
plt2 = plt.imshow(X_test[100])
plt.title(y_test[100])
print("==================================================================================")
print()
print("Class Name - ",y_test[100])
print()
plt.show()
print("==================================================================================")