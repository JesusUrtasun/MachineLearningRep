# TensorFlow Chapter 4. Overfitting and underfitting
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Check the version of TensorFlow
print("TensorFlow version: {}".format(tf.__version__))
print("TensorFlow Chapter 4: Overfitting and underfitting")

# As seen in previous chapters, accuracy on the validation set peaks at a particular number of epochs, and the starts decreasing
# Overfitting has happened. The network learns patterns on the train set that do not generalize to the test data
# Prevent overfitting by use more training data. When not possible, use regularization techniques
# Download the Internet Movie Database IMDB. Multi-hot encode the sentences (turning them into vectors of 0s and 1s)
# Example, the sequence [3, 5] will be a 10000-dim vector with all zeros except for the indices 3 and 5, being there ones
NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words = NUM_WORDS)
def multi_hot_sequences(sequences, dimension = NUM_WORDS):
    # Create an zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        # Set specific indices of results[i] to be 1
        results[i, word_indices] = 1.0
    return results

train_data = multi_hot_sequences(train_data)
test_data = multi_hot_sequences(test_data)
# Look at the multi-hot encoded vectors. Word indices are sorted by frequency, so there are more one-values near index zero
print("Example. 1st element of the training set:")
plt.plot(train_data[0])
plt.show()

# Demonstrate overfitting.
# Simplest way to avoid it is by reducing the model. Then, there are less parameters to learn. Number of parameters to learn, "capacity"
# Recall, deep learning models are good at fitting to the training data, but real goal is generalization, not fitting
# Start by building a simple model with only Dense layers
print("\n1. Build baseline model")
baseline_model = keras.Sequential([
    # input shape is required only required so that .summary() works
    keras.layers.Dense(16, activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
    keras.layers.Dense(16, activation = tf.nn.relu),
    keras.layers.Dense(1, activation = tf.nn.sigmoid)
    ])
baseline_model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
baseline_model.summary()
print("\nTrain the baseline model")
baseline_history = baseline_model.fit(train_data, train_labels,
    epochs = 20, batch_size = 512, validation_data = (test_data, test_labels), verbose = 1)

# Build a bigger model
print("\n2. Build bigger model")
bigger_model = keras.Sequential([
    # input shape is required only required so that .summary() works
    keras.layers.Dense(256, activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
    keras.layers.Dense(256, activation = tf.nn.relu),
    keras.layers.Dense(1, activation = tf.nn.sigmoid)
    ])
bigger_model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
bigger_model.summary()
print("\n Train the bigger model")
bigger_history = bigger_model.fit(train_data, train_labels,
    epochs = 20, batch_size = 512, validation_data = (test_data, test_labels), verbose = 1)

# Plot the training and validation loss
def plot_history(histories, key = "binary_crossentropy"):
    plt.figure(figsize = (16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history["val_"+key], "--", label = name.title() + "Val")
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(), label = name.title() + "Train")

    plt.xlabel("Epochs")
    plt.ylabel(key.replace("_", " ").title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

print("\n3. Plot the training and validation loss for every model")
plot_history([("baseline", baseline_history), ("bigger", bigger_history)])
plt.show()
# The larger network begins overfitting almost right away, after just one epoch and overfits much more severely
# The more capacity a network has, the quicker it will be able to model the training data (low training loss)

# Strategies.
# Weight regularization
print("\n5. Add weight regularization")
# Occam's priciple. There are multiple sets of weights that can fit the data, and simple models are less likely to overfit
# Simple model, where the distribution of parameter values has less entropy.
# Force the weights to take small values, making the distribution more "regular"
# Add to the loss function a cost associated with having large weights. L1 and L2 regularization
L2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001),
        activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001),
        activation = tf.nn.relu),
    keras.layers.Dense(1, activation = tf.nn.sigmoid)
])
L2_model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
L2_history = L2_model.fit(train_data, train_labels,
    epochs = 20, batch_size = 512, validation_data = (test_data, test_labels), verbose = 1)
print("\nPlot loss after weight regularization")
plot_history([("baseline", baseline_history), ("L2", L2_history)])
plt.show()

# Dropout
print("\n6. Add dropout")
# Applied to a layer, randomly "dropping out" a number of output features of the layer during training
# [0.2, 0.5, 1.3, 8, 1, 1] will become [0, 0.5, 1.3, 0, 1, 1]. Dropout rate usually from 0.2 to 0.5
# At test time no values are dropped out. Instead, the output values are scaled down by a factor equal to the dropout rate
# This balances the fact that now more units are active that during the training
Dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation = tf.nn.relu, input_shape = (NUM_WORDS,)), 
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = tf.nn.sigmoid)
])
Dpt_model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
Dpt_history = L2_model.fit(train_data, train_labels,
    epochs = 20, batch_size = 512, validation_data = (test_data, test_labels), verbose = 1)
print("\nPlot loss after weight regularization")
plot_history([("baseline", baseline_history), ("Dropout", Dpt_history)])
plt.show()
