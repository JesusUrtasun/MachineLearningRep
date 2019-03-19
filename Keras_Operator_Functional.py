# Implement operators in Keras. Functional API model
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input
import matplotlib.pyplot as plt

# Class ConvLayer performing the convolution
class MyConv(keras.layers.Layer):

    # Init method. Receive as input a numpy array and convert it to a keras tensor
    def __init__(self, output_dim, mat = None, **kwargs):
        self.output_dim = output_dim
        self.matrix_shape = mat.shape
        # If transpose, output_axis = 0 / If no transpose, output_axis = -1
        self.output_axis = 0
        self.mat = K.tf.transpose(K.constant(mat), perm = [0, 1, 2, 3])
        # Let the abstract class handle the **kwargs positional arguments
        super(MyConv, self).__init__(**kwargs)
    
    # Build method
    def build(self, input_shape):
        super(MyConv, self).build(input_shape)
    
    # Call method
    def call(self, x):
        # Tensor product with keras
        # return K.dot(x, self.mat)
        # Tensor product with tensorflow
        return K.tf.tensordot(x, self.mat, axes = 2)

    # Compute_output_shape method
    def compute_output_shape(self, input_shape):
        # Output shape must be the last index of the matrix to convolute with
        # Default, matrix to convolute has dimensions (n, m)
        #return (input_shape[0], self.matrix_shape[self.output_axis])
        # More dimensions in the matrix to convolute
        return (self.matrix_shape[2], self.matrix_shape[3])

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

# Reshape all images and save them in the x_train and x_test[] lists
x_train = []
x_test = []
for x_vector in x_train_raw:
    x_train.append(x_vector.reshape(784))
for x_vector in x_test_raw:
    x_test.append(x_vector.reshape(784, 1))
print("Reshape\nTrain set, example 0: {}".format(x_train[0].shape))
print("Train labels example: {}".format(y_train_raw[0].shape))

# Playing with tensors
print("\nPlaying with tensors")

# Play with permutations
x1 = K.tf.constant([[1, 2, 3], [4, 5, 6]])
print("x1 = {}".format(x1))
x1_trans = K.tf.transpose(x1, perm = [1, 0])
print("x1 = {}".format(x1_trans))
x2 = K.tf.constant([[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]])
print("x2 = {}".format(x2))
x2_trans = K.tf.transpose(x2, perm = [0, 2, 1])
print("x2 = {}".format(x2_trans))

# Play with tensordot
a = K.tf.constant([[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]])
a_np = np.ones((2, 2, 3))
print("a = {}".format(a))
print("a_np = {}".format(a_np.shape))
b = K.tf.constant( np.array([[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]], dtype = np.int32).T)
b_np = np.ones((2, 2, 3))
print("b = {}".format(b))
print("b_np = {}".format(b_np.shape))
c_np = np.tensordot(a_np, b_np, axes = ([1,0], [1,0]))
print("c_np = {}".format(c_np.shape))
c = K.tf.tensordot(a, b, axes = 1)
print("c = {}".format(c))
c = K.tf.tensordot(a, b, axes = 2)
print("c = {}".format(c))

# Toy matrix to convolute with. Extend later to higher-dimensional tensor
#my_matrix = np.ones((784, 64, 12, 10))
my_matrix = np.ones((784, 64, 12, 10))

# Keras functional API
print("\nKeras Functional model")
print("Building layers")
l1 = Dense(64, activation = "relu")
lconv = MyConv(10, mat = (my_matrix))
l2 = Dense(10, activation = "softmax")
# Functional model requires a list of inputs and a list of outputs
# list_inputs will have all input layers in parallel
# list_outputs will have the l2 action on the previous layers
list_outputs = []
list_inputs = []
for x_array in x_train[:10]:
    l0 = Input(tensor = K.constant(x_array.reshape(-1,1)))
    list_inputs.append(l0)
    list_outputs.append(l2(lconv(l1(l0))))
# Convert the labels in the train and testing set into one-hot encoded vectors
y_train = []
y_test = []
for y_label in y_train_raw[:10]:
    y_label_hot = keras.utils.to_categorical(y_train_raw[y_label], num_classes = 10)
    y_train.append(np.array([y_label_hot]))
for y_label in y_train_raw[:10]:
    y_label_hot = keras.utils.to_categorical(y_test_raw[y_label], num_classes = 10)
    y_test.append(np.array([y_label_hot]))

print("Building model")
model = Model(list_inputs, list_outputs)
model.summary()
print("model output" + str(model.output_shape))

print("Compiling")
model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

print("Fitting")
model.fit(x = None, y = y_train, epochs = 3)

# Build a second model that reuses the layers, to evaluate accuracy and make predictions

# model_test 
# model_test.add(l1)
# model_test.add(l2)
# model_test.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
# print("Evaluating") 
# test_loss, test_acc = model_test.evaluate(x_test[0].T, y_test[0], batch_size=1)
# print("Test loss = {}\nTest accuracy = {}".format(test_loss, test_acc))

# print("Making predictions")
# predictions = model_test.predict(np.array(x_test))
# # Predictions is an array of ten numbers from 0 to 1
# number_test = int(input("Choose one review of the 1000 in the testing set: "))
# plt.figure()
# plt.imshow(x_test[number_test].reshape(28,28))
# plt.show()
# # Use the argmax to pick the element with the highest value
# print("Prediction: {}".format(np.argmax(predictions[number_test])))
# print("Truth: {}".format(y_test_raw[number_test]))