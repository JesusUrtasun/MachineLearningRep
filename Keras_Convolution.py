# Implement convolution in Keras. Following the NNPDF structure
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input

# Class ConvLayer performing the convolution
class MyConv(keras.layers.Layer):

    # Init method. Receive as input a numpy array and converts it to a keras tensor
    def __init__(self, output_dim, mat = None, **kwargs):
        self.output_dim = output_dim
        self.matrix_shape = mat.shape
        # If transpose, output_axis = 0 / If no transpose, output_axis = -1
        self.output_axis = 0
        self.mat = K.tf.transpose(K.constant(mat), perm = [0, 2, 1, 3])
        # Let the abstract class handle the **kwargs positional arguments
        super(MyConv, self).__init__(**kwargs)
    
    # Build method
    def build(self, input_shape):
        super(MyConv, self).build(input_shape)
    
    # Call method
    def call(self, x):
        # Tensor product with Keras.tensorflow.tensordot
        return K.tf.tensordot(x, self.mat, axes = 2)

    # Compute_output_shape method
    def compute_output_shape(self, input_shape):
        # Output shape must be the last indeces of the matrix to convolute with. Now (k, l, m, n)
        return (self.matrix_shape[1], self.matrix_shape[3])

# Function performing a comparison
def Compare(tensor_set, my_model1, my_model2, my_option = 0):
    # Prediction of model1 gives already a y vector to match with data
    y1 = my_model1.predict(x = None, steps = 1)
    # Prediction of model2 gives a pdf. Perform manually the convolution
    pdf2 = my_model2.predict(x = None, steps = 1)
    y2 = np.tensordot(pdf2[my_option], tensor_set[my_option], axes = [[0, 1], [0, 2]])
    # Compute comparison
    ratio = y1[my_option] / y2
    print("Ratio y1_set[{}] / y2_set[{}] = {}".format(my_option, my_option, ratio))

### Main ###

# Generate train and data sets
print("\nGenerating train and data sets:")
x1 = np.random.random((40, 1))
x2 = np.random.random((30, 1))
x3 = np.random.random((25, 1))
print("x1 = {}".format(x1.shape))
print("x2 = {}".format(x2.shape))
print("x3 = {}".format(x3.shape))
y1_data = np.random.random((1, 131))
y2_data = np.random.random((1, 125))
y3_data = np.random.random((1, 128))
print("y1_data = {}".format(y1_data.shape))
print("y2_data = {}".format(y2_data.shape))
print("y3_data = {}".format(y3_data.shape))
# Functional method requires a list of inputs and a list of outputs
x_train = [x1, x2, x3]
y_data = [y1_data, y2_data, y3_data]

# Matrix to convolute with
fk_1 = np.random.random((40, 1, 14, 131))
fk_2 = np.random.random((30, 1, 14, 125))
fk_3 = np.random.random((25, 1, 14, 128))
fk_set = [fk_1, fk_2, fk_3]

# Functional model performing the convolution
print("\nModel: 1")
l1 = Dense(14, activation = "relu")
lconv = [MyConv(131, mat = fk_1), MyConv(131, mat = fk_2), MyConv(128, mat = fk_3)]

# First model, performs the convolution
list_inputs = []
list_outputs = []
for i, xi in enumerate(x_train):
    x_tensor = K.constant(xi)
    l0 = Input(tensor = x_tensor)
    list_inputs.append(l0)
    list_outputs.append(lconv[i](l1(l0)))
print("Building model")
model1 = Model(list_inputs, list_outputs)
model1.summary()
print("Compiling")
model1.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
print("Fitting")
model1.fit(x = None, y = y_data, epochs = 3, batch_size = 3)

# Second model, with PDF set as output
print("\nModel: 2")
list_outputs2 = []
for i in list_inputs:
    list_outputs2.append(l1(i))
print("Building model")
model2 = Model(list_inputs, list_outputs2)
model2.summary()
print("Compiling")
model2.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

# Performing a comparison
print("\nComparing")
Compare(fk_set, model1, model2)
