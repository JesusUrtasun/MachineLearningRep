# TensorFlow Chapter 3. Regression
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Check the version of TensorFlow
print("TensorFlow version: {}".format(tf.__version__))
print("TensorFlow Chapter 3: Regression")

# Regression. Predict the output of a continuous value, like a price or a probability
# Download the Auto MPG Dataset , and build a model to predict the fuel efficiency of late 1970 - 1980
# Describe automoviles with atributes like: cylinders, displacement, horsepower, and weight
dataset_path = keras.utils.get_file("auto_mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# Import it using pandas
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", sep = " ", comment = "\t", skipinitialspace = True)
dataset = raw_dataset.copy()
print("Example. Last 5 elements of the dataset: {}".format(dataset.tail(5)))

# Clean the data. The dataset contains unknown values
print("\nExample. Dropping the unknown values:\n{}".format(dataset.isna().sum()))
# For this initial case, drop these rows
dataset = dataset.dropna()
print("\nDataset:\n{}".format(dataset))
print("Data structure: {}".format(dataset.shape))
# The Origin column is categorical, not numerical. Convert it to a one - hot encoded
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0
dataset.tail()
print("\nDataset:\n{}".format(dataset))
print("Data structure: {}".format(dataset.shape))

# Split the data into a training set and a test set
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

# Explore the data. Plot the distributions of a few pairs of colums from the training set
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind = "kde")
# Also look at the overall statistics
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print("Train stats:\n{}".format(train_stats))

# Separate the target value, or "label", from the features. This label will be used to feed the NN
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

# Normalize the data. Ranges of different features are extremely different
def norm(x):
    return(x - train_stats["mean"]) / (train_stats["std"])
norm_train_data = norm(train_dataset)
norm_test_data = norm(test_dataset)

# Build the model
print("\n1. Build and compile the model")
# Use a Sequential model with two densely connected hidden layers, and output given a single continous value.
# Model building steps ara wrapped in a function, build_model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_dataset.keys())]),
        layers.Dense(64, activation = tf.nn.relu),
        layers.Dense(1)])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss = "mse", optimizer = optimizer, metrics = ["mae", "mse"])
    return model

model = build_model()

# Inspect the model
print("\n2. Model structure")
model.summary()
# Try the model. Take a batch of 10 examples from the training data and call model.predict()
example_batch = norm_train_data[:10]
example_result = model.predict(example_batch)
print("Example result:\n{}".format(example_result))

# Train the model
print("\n3. Train the model")
# Train the model in 1000 epochs, and record the training and validation accuracy in the history object
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print("")
        print(".", end = "")

number_epochs = 1000
print("Start training:")
history = model.fit(
    norm_train_data, train_labels, 
    epochs = number_epochs, validation_split = 0.2,
    verbose = 0, callbacks = [PrintDot()])

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print("\nVisualize the model with hist.tail():\n{}".format(hist.tail()))

# Plot the errors vs the epoch of the training and validation sets
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    plt.show()

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Mean Abs Error [MPG]")
plt.plot(hist["epoch"], hist["mean_absolute_error"], label = "Train error")
plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label = "Val error")
plt.legend()
plt.ylim([0, 5])

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error [$MPG^2$]")
plt.plot(hist["epoch"], hist["mean_squared_error"], label = "Train error")
plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label = "Val error")
plt.legend()
plt.ylim([0, 20])

print("Plot history atributes")
plot_history(history)

# Plot shows little improvement, or even degradation after 100 epochs.
# Let's update the model.fit() to automatically stop training when the validation score does't improve
model = build_model()
# Patience parameter is the amount of epochs to check for an improvement
early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10)
history2 = model.fit(norm_train_data, train_labels,
    epochs = number_epochs, validation_split = 0.2,
    verbose = 0, callbacks = [early_stop, PrintDot()])

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Mean Abs Error [MPG]")
plt.plot(hist["epoch"], hist["mean_absolute_error"], label = "Train error")
plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label = "Val error")
plt.legend()
plt.ylim([0, 5])

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error [$MPG^2$]")
plt.plot(hist["epoch"], hist["mean_squared_error"], label = "Train error")
plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label = "Val error")
plt.legend()
plt.ylim([0, 20])

print("\nPlot history atributes after the EarlyStopping")
plot_history(history2)

# Make predicitions
print("\n4. Make predictions")
predictions = model.predict(norm_test_data).flatten()
# Plot predictions vs true values
plt.scatter(test_labels, predictions)
plt.xlabel("True values [MPG]")
plt.ylabel("Predictions [MPG]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()
# Plot the error distribution
error = predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()

# Conclusion
# 1. Mean Squared Error (MSE) is a common loss function used in regression problems
# 2. Evaluation metrics used for regression differ from classification. Common regression metric is Mean Absolute Error (MAE)
# 3. When numeric input data have values with different scales, each feature should be scaled independently to the same range
# 4. When there is not much training data, use a small network to a few hidden layers to avoid overfitting.
# 5. Early stop is a usefull technique to prefent overfitting.