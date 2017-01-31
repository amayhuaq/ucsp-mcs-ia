# ucsp-mcs-ia-tensorflow-cnn
Creating a convolutional neural network using Tensorflow and Python.

This code was tested with the dataset CIFAR-10: 

<http://www.cs.utoronto.ca/~kriz/cifar.html>

This dataset has:
- Training data: 50000 images (5 data files)
- Testing data: 10000 images (1 data file)

Some details about configuration of this CNN are:
- Based on CPU
- Image size: 32 x 32
- Image size cropped: 24 x 24
- Batch size training: 64
- Batch size testing: 256
- Optimizer: tf.train.AdamOptimizer
- Learning_rate: 1e-4
- Configuration of model:

```python

# Configuration of CNN model: creating tensors

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

########################################################
# Creating neural network for training phase

# Batch size to the training set
train_batch_size = 64
# Number of optimization iterations
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
# Create the neural network to be used for training
_, loss = create_network(training=True)
# Optimizer which will minimize the loss function
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

########################################################
# Creating neural network for test phase

# Batch size to the testing set
batch_size = 256
# In this phase, we only need y_pred that is the predicted class-labels
y_pred, _ = create_network(training=False)
# We calculate the predicted class-number that is the index of the largest element in the array y_pred
y_pred_cls = tf.argmax(y_pred, dimension=1)
# We create a vector of booleans that tells if the predicted class is equal the true class
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# We calculate the classification accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

```

About the number of iterations executed, we obtain the following results:

- 500 iterations: Accuracy on Test-Set: 38.6% (3861 / 10000)
- 2500 iterations: Accuracy on Test-Set: 49.8% (4978 / 10000)
- 5000 iterations: Accuracy on Test-Set: 
- 10000 iterations: Accuracy on Test-Set: 

