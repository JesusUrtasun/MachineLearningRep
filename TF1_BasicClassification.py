# TensorFlow Chapter 1. Basic Classification
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Check the version of TensorFlow
print("TensorFlow version: {}".format(tf.__version__))
print("TensorFlow Chapter 1: Basic Classification")

# Download the Fasion MNIST dataset. 70000 images of clothing articles of 28x28 pixels
# Use 60000 to train the network and 10000 to test it
# Load the data will return 4 numpy arrays: train images, train labels, test images, test labels
# Images are 28x28 numpy arrays with pixel values from 0 to 255. Labels are arrays of integers from 0 to 9
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Store the class names, mapping from the numbers 0-9
class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Angle boot"]

# Explore the data
print("Train images and labels structure:")
print("Train images: {}".format(train_images.shape))
print("Train labels: {}".format(train_labels.shape))
print("Test images and labels structure:")
print("Test images: {}".format(test_images.shape))
print("Test labels: {}".format(test_labels.shape))

# Process the data before training
print("Example: 1st element of the training images")
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.show()
# Scale the values before training, to be form 0 to 1
train_images = train_images / 255
test_images = test_images / 255

# Display the first 25 images
print("Example: 1st 25 elements of the training images together with labels")
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # Set location to all labels
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model. Start with a Multiple Layer Peceptron MLP, many layers chained together
print("\n1. Build the NN model. Multi Layer Peceptron MLP with 1 hidden layer")
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)])
# First layer tf.keras.Flatten transforms images from a 2d-array (28 by 28 in this case) to a 1d-array 28 * 28 = 784 pixels
# Hidden layers tf.keras.Dense() are basic layer with a given number of nodes and an activation function
# Output layer tf.keras.Dense() is a 10-node layer with Softmax as activation function

# Compile the model
print("\n2. Compile the NN model")
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
# Optimizer sets how the model is updated based on the data and its loss function
# Loss function measures the degree of accuracy during trainig
# Metrics, monitoring training and test steps. Use now accurcay, fraction of images correctly classified

# Train the model
print("\n3. Train the NN model")
model.fit(train_images, train_labels, epochs = 3)
# 1. Feed the model with the training data. Now, the train_images and train_labels
# 2. The model learns to associate images and labels
# 3. Ask the model to give predictions about the test set. Verify if predictions match the labels in the test_labels array

# Evaluate accuracy
print("\n4. Evaluate accuracy")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: {}".format(test_acc))

# Make predictions
# With the trained model, predict a label for each image on the testing set
print("\n.5 Make predictions")
predictions = model.predict(test_images)
# Predictions is an array of ten numbers from 0 to 1. The argmax picks the element with the higher value
# Process the data before training
print("What do you thing it is?")
print("Prediction: {}, {}".format(np.argmax(predictions[0]), class_names[np.argmax(predictions[0])]))
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.show()
print("Truth: {}, {}".format(test_labels[0], class_names[test_labels[0]]))


