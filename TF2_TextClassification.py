# TensorFlow Chapter 2. Text Classification
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Check the version of TensorFlow
print("TensorFlow version: {}".format(tf.__version__))
print("TensorFlow Chapter 2: Text Classification")

# Binary classification. Classify reviews in positive or negative looking at the words used
# Download the Internet Movie Database IMDB. 25000 movie reviews for training and 25000 for testing. Both sets ara balanced
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
# Argument num_words = 10000 keeps the 10000 most frequently used words

# Explore the data. Each elements contains a list of numbers corresponding to a word in the dictionary
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print("Example. 1st element of the training set:\n{}".format(train_data[0]))
# Different movie reviews could have different shape. 
print("\nExample. Number of words of the 3 1st reviews: {}, {}, {}".format(len(train_data[0]), len(train_data[1]), len(train_data[2])))

# Convert the integers back to words. Create a function to call a dictionary containing the integer to string mapping
print("Integer - word mapping function")
word_index = imdb.get_word_index()
# The first indices are reserved
for k,v in word_index.items():
    word_index[k] = v+3
# Alternative, using dictionary instead of list
# word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print("\nExample. 1st review (decoded):\n{}".format(decode_review(train_data[0])))

# Prepare the data. Each review needs to be converted to a tensor before feed the NN
# Pad the arrays so they all have the same length, using pad_sequences function.
# Then create an integer tensor of shape max_lengt * num_reviews
# We can use an embedding layer capable of handling this shape as the first layer in our network
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 265)
test_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 265)
print("Example: 1st element of the training set after padding:\n{}".format(train_data[0]))
print("Example: Number of words of the 3 1st reviews: {}, {}, {}".format(len(train_data[0]), len(train_data[1]), len(train_data[2])))

# Build the model
print("\n1.Build the NN model")
# Input shape is the vocabulary amount used for the reviews, 10000 words 
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = tf.nn.relu))
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid))
model.summary()
# First layer is an embedding layer. Takes integer - encoded vocabulary and looks up the embedding vector for each word index
# Second layer, GlobalAveragePooling1D, returns a fixed length output vector for each sample by averaging over the sequence dimension,
# Fixed length output vector is piped through a fully connected Dense layer with 16 nodes
# Output layer, densely connected with a single output node with sigmoid as activation function

# Compile the model
print("\n2. Compile the NN model")
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
# Create a validation set by setting apart 10000 examples from the original training data, to train an validate only using the training set
# After the training and validation, we will proceed to give predictions about the testing set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
print("\n3. Train the NN model")
history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512, validation_data = (x_val, y_val), verbose = 1)
# Train for 30 epochs in sets of 512 samples. While training, monitore the model's loss and accuracy on the 10000 reviews ofvalidation set

# Evaluate the model
print("\n4. Evaluate accuracy")
results = model.evaluate(test_data, test_labels)

# Create a graph. model.fit() returns a History object that contains a dictionary with everything happened during the training
history_dict = history.history
history_dict.keys()
acc = history_dict["acc"]
val_acc = history_dict["val_acc"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)

# Build the Loss graph. "bo" means blue dots, and "b" blue line
plt.plot(epochs, loss, "bo", label = "Training loss")
plt.plot(epochs, val_loss, "b", label = "Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# Build the Accuracy graph.
plt.plot(epochs, acc, "bo", label = "Training loss")
plt.plot(epochs, val_acc, "b", label = "Validation loss")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# Check, loss and accuracy evolve with time, loss decreasing and accuracy increasing
# Validation loss and accuracy stabilize from about 20 epochs, due to OVERFITTING. The model performs better on training than on new data.
# Prevent the overfitting by stopping the traiing at 20 epochs.

# Make predictions
print("\n5. Make predictions")
predictions = model.predict(test_data)
number_test = int(input("Choose one review of the 2500 in the testing set: "))
print("Review number {}:\n{}".format(number_test, decode_review(test_data[number_test])))
if predictions[number_test] > 0.5:
    print("Prediction: Positive")
else:
    print("Prediction: Negative")
if test_labels[number_test] > 0.5:
    print("Truth: Positive")
else:
    print("Truth: Negative")
