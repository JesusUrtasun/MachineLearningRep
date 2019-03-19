# Implement operators in Keras. Sequential model
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape
import matplotlib.pyplot as plt

# Class ConstantLayer
class ConstantLayer(keras.layers.Layer):

    # Init method. Set self.build = True at the end, by calling super([Layer], self).build()
    def __init__(self, output_dim, const = 1, **kwargs):
        self.output_dim = output_dim
        self.const = const
        # Let the abstract class (super of MyLayer) handle the **kwargs positional arguments
        super(ConstantLayer, self).__init__(**kwargs)
    
    # Build method. Define our weights
    def build(self, input_shape):
        # Create a trainable weight variable
        self.my_weights = self.add_weight(name = "my_weights", shape = (input_shape[1], self.output_dim), initializer = "uniform", trainable = True)
        self.my_bias = self.add_weight("my_bias", shape = (self.output_dim,), initializer = "uniform", trainable = True)
        # Be sure to call this at the end
        super(ConstantLayer, self).build(input_shape)
    
    # Call method.
    def call(self, x):
        return K.dot(x, self.my_weights) + self.my_bias + self.const

    # Compute_output_shape method. 
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# Class ReshapeLayer
class ReshapeLayer(keras.layers.Layer):

    # Init method. Set self.build = True at the end, by calling super([Layer], self).build()
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        # Let the abstract class (super of MyLayer) handle the **kwargs positional arguments
        super(ReshapeLayer, self).__init__(**kwargs)
    
    # Build method. No weights are needed for this layer
    def build(self, input_shape):
        # Be sure to call this at the end
        super(ReshapeLayer, self).build(input_shape)
    
    # Call method.
    def call(self, x):
        # Next step, reshape without the keras reshape function
        return K.reshape(x, (-1, self.output_dim))

    # Compute_output_shape method. 
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

### Main ###

# Download datasets
print("\nDownload datasets")
mnist = keras.datasets.mnist
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()
# Explore the data
print("Train set: {}".format(x_train_raw.shape))
print("Train labels: {}".format(y_train_raw.shape))
print("Test set: {}".format(x_test_raw.shape))
print("Test labels: {}".format(y_test_raw.shape))
# Scale the values before training, to be form 0 to 1
x_train_raw = x_train_raw / 255
x_test_raw = x_test_raw / 255
x_train = []
x_test = []

# Reshape all images and save them in the x_train and x_test lists
for x_vector in x_train_raw:
    x_train.append(x_vector.reshape(784))
for x_vector in x_test_raw:
    x_test.append(x_vector.reshape(784))
print("Reshape\nReshaped Train set, example 0: {}".format(x_train[0].shape))
print("Train labels example: {}".format(y_train_raw[0].shape))

# Keras Sequential model
print("\nKeras Sequential model")
print("\nBuilding model")
model = Sequential()
model.add(Dense(64, activation = "relu", input_shape = (784,)))
# Add Reshape layer, from a vector of 64 elements to a matrix of nx4
model.add(Reshape((-1, 4), input_shape = (64,)))
# Add manual ReshapeLayer, reshaping from the matrix to the original vector size
model.add(ReshapeLayer(64))
# Next step. Try to invert the order of the reshape
model.add(ConstantLayer(32, const = 2))
model.add(Dense(10, activation = "softmax",))
print("Dense2 = {}".format(model.output_shape))

print("model output" + str(model.output_shape))
model.summary()

print("Compiling")
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

print("Fitting")
model.fit(x = np.array(x_train), y = y_train_raw, epochs = 3)

print("Evaluating") 
test_loss, test_acc = model.evaluate(np.array(x_test), y_test_raw)
print("Test loss = {}\nTest accuracy = {}".format(test_loss, test_acc))

print("Making predictions")
predictions = model.predict(np.array(x_test))
# Predictions is an array of ten numbers from 0 to 1
number_test = int(input("Choose one review of the 1000 in the testing set: "))
plt.figure()
plt.imshow(x_test[number_test].reshape(28,28))
plt.show()
# Use the argmax to pick the element with the highest value
print("Prediction: {}".format(np.argmax(predictions[number_test])))
print("Truth: {}".format(y_test_raw[number_test]))