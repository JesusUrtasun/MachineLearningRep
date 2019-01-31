import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scp
from sklearn import datasets
from sklearn.utils import shuffle

# Function definitons
def Sigmoid(x):
    return scp.expit(x)

def derSigmoid(x):
    return Sigmoid(x)*(1-Sigmoid(x))

class NeuralNetwork:
    
    def __init__(self, example_in, example_out, nlayers = 4, nodes = 128, learning = 0.1):

        self.input_size = len(example_in)
        self.output_size = len(example_out)
        self.nlayers = nlayers
        self.nodes = nodes
        self.learning = learning
        self.w = []
        self.bias = []

        # Neural Network structure
        print ("\nInitialize Neural Network")
        
        # Iterate over the number of layers and set the nodes per layer
        k = self.nlayers - 2
        list_nodes = [len(example_in)]
        for _ in range(k):
            # Hidden layers will have all same number of nodes
            input_nodes = self.nodes
            list_nodes.append(input_nodes)
        # Output layer will have as much nodes as the output example
        input_nodes = len(example_out)
        list_nodes.append(input_nodes)
        print("NN structure: {0}".format(list_nodes))

        # Initialize weights and biases with random numbers btw -0.5 and 0.5
        for i in range(self.nlayers - 1):
            self.w.append( np.random.rand(list_nodes[i+1], list_nodes[i]) - 0.5 ) 
            self.bias.append( np.random.rand(list_nodes[i+1]) - 0.5 )

    # Forward feeding
    def ForwardFeeding(self, x):
        
        # Make sure x is an array
        x_in = np.array(x)
        z = [0]
        a = [x_in]
        # Iterate over the number of layers filling z, a for each one
        for i in range(self.nlayers - 1):
            z.append(np.dot(self.w[i], a[-1]) + self.bias[i])
            a.append(Sigmoid(z[-1]))
        
        return z, a

    # Back propagation
    def BackPropagation(self, z, a, target):
        
        # Compute delta and dW from the output layer by comparing with the example
        delta = (target - a[-1]) * derSigmoid(z[-1])
        dw = np.outer(delta, a[-2])
        dw = np.outer(delta, a[-2])
        updateWeights = [(delta, dw)]
        # Run backwards from the output layer
        for i, w_i in enumerate(reversed(self.w[1:])):
            delta = np.dot(delta, w_i) * derSigmoid(z[-2-i])
            dw = np.outer(delta, a[-3-i])
            updateWeights.append((delta, dw))

        # Reverse the updated list of weights and biases
        updateWeights.reverse() 

        # Update the lists of weights and biases
        for i in range(len(self.w)):
            self.w[i] += self.learning * updateWeights[i][1]
            self.bias[i] += self.learning * updateWeights[i][0]
        
        return self.w, self.bias

    # Cost function
    def CostFunction(self, y, target):

        # Compare element by element the output layer with the example
        a = (y-target)
        a2 = np.dot(a, a)/len(a)

        return a2

    # Training the Neural Network
    def Train(self, set_in, set_target):
        
        # Run for each element in the examples in the input set, comparing always with the example
        for x, y in zip(set_in, set_target):
            z, a = self.ForwardFeeding(x)
            w, bias = self.BackPropagation(z, a, y)

    # Training the Neural Network
    def Test(self, set_in, set_target):
        
        # Define success
        c = 0.0
        success = 0.0
        for x, y in zip(set_in, set_target):
            _, a = self.ForwardFeeding(x)
            c += self.CostFunction(a[-1], y)
            # Accumulate one succes if prediction and example match
            if a[-1].argmax() == y.argmax():
                success += 1.0
        
        c = c/len(set_in)
        succes = success/len(set_in)

        return c, succes
    
    def TrainWrapper(self, set_in, set_target, n_train, gen_test = 0.1):

        total_len = len(set_in)
        test_len = int(total_len * gen_test)

        # Test sets from the dataset. [a:b] notation for [a,b), from -test_len to the end
        test_x = set_in[-test_len:]
        test_y = set_target[-test_len:]

        # Train sets from the dataset
        train_x = set_in[:-test_len]
        train_y = set_target[:-test_len]

        # Shuffle the train sets before training
        for i in range(n_train):
            set_x, set_y = shuffle(train_x, train_y)
            self.Train(set_x, set_y)
            print("\nTraning number {0} finished".format(i+1))
            print("Computing test")
            ave_cost, ratio = self.Test(test_x, test_y)
            print("The average error was: {0}".format(ave_cost))
            print("With success ratio of: {0}".format(ratio))        


# Main.

# Download datasets
print("Download datasets")
data = datasets.load_digits()
x_set_full = data["data"]/16
y_set_full = np.eye(10)[data["target"]]
# x_set contains 1797 arrays, of 64 elemens (input values) each
# x_set contains 1797 arrays, of 10 elemens (output values) each
print("x = {0}\ny = {1}".format(x_set_full.shape, y_set_full.shape))

# Generate two lists containing all x and y sets respectively
x_set = []
y_set = []
for x, y in zip(x_set_full, y_set_full):
    x_set.append(x)
    y_set.append(y)
# print("x = {0}\ny = {1}".format(x_set[0], y_set[0]))

# Define a particular example to plot and check
example_n = 271
example_in = x_set[example_n]
example_out = y_set[example_n]
# print("Input = {0}\n Output = {1}".format(example_in, example_out))
# Reshape the input elements form 64-dim array to (8,8) matrices and plot it
image_x = example_in.reshape(8,8)
plt.imshow(image_x)
plt.show()

# Instance the Neural Network class with the given inputs
NN = NeuralNetwork(example_in, example_out, nlayers = 5)
# Call the ForwardFeeding method of the Neural Network class
a, z = NN.ForwardFeeding(example_in)
# print("Example in: {0}".format(a[-1]))
# Call the TrainWrapper method of the Neural Network class
NN.TrainWrapper(x_set, y_set, n_train = 5)
# Call the Test method to test with example_in
z, a = NN.ForwardFeeding(example_in)
class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
print("Prediction = {0}, True = {1}".format(np.argmax(np.array(a[-1])), np.argmax(example_out)))