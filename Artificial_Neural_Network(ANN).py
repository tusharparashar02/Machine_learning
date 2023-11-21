# single level perceptron

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the data

(train_images, train_labels), (test_images, test_lables) = mnist.load_data()

# Flatten the images into 1D array

train_images = train_images.reshape((60000, 28*28)).astype('float32')/255

test_images = test_images.reshape((10000, 28*28)).astype('float32')/255

#  one hot encode the labels

train_labels = to_categorical(train_labels)
test_lables = to_categorical(test_lables)

# Build the single layer preceptron model

model = models.Sequential()
model.add(layers.Dense(10, activation='softmax', input_shape=(28*28,)))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=8, validation_data=(test_images, test_lables))

# Evaluate the model

test_loss, test_acc = model.evaluate(test_images, test_lables)
print(f'Test accuracy: {test_acc * 100:.2f}%')


from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict(test_images)
predictions_lables = np.argmax(predictions, axis=1)
true_lables = np.argmax(test_lables, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(true_lables, predictions_lables)

# plot confusion matrix

plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()

classes = [str(i) for i in range(10)]
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Classification report
report = classification_report(true_lables, predictions_lables, target_names=classes)
print("Classification report: \n", report)
