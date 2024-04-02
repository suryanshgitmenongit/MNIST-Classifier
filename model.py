import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

for i in range(9):
    plt.subplot(3, 3, i+1)
    num = random.randint(0, len(X_train))
    plt.imshow(X_train[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[num]))

plt.tight_layout()

# Flatten the input data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Convert to float32 and scale the pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# One-hot encode the target labels
no_classes = 10
Y_train = to_categorical(y_train, no_classes)
Y_test = to_categorical(y_test, no_classes)

# Define the model architecture
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1)

# Evaluate the model on the test data
score = model.evaluate(X_test, Y_test)
print('Test accuracy:', score[1])

# Make predictions on the test set
results = model.predict(X_test)
predicted_classes = np.argmax(results, axis=1)

# Plot correct and incorrect predictions
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()

for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

plt.tight_layout()

plt.figure()

for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

plt.tight_layout()

# Calculate and print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, predicted_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted_classes))

# Calculate and print F1 score
f1 = f1_score(y_test, predicted_classes, average='weighted')
print("F1 Score:", f1)
