import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

import plot_functions
import load_cifar10

# Global variables
data = None
class_names = None
session = None
LEARNING_RATE = 1e-4

# Image configuration
from load_cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24

"""
This function adds nodes to the graph that pre-process a random image and create
random variations of the original input images.
This function takes a single image as input and a boolean whether to build the training or testing graph.
"""
def pre_process_image(image, training):    
    # For training, add the following to the TensorFlow graph.
    if training:
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For testing, crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image


"""
This function loops over all the input images and call the function 
above which takes a single image as input.
"""
def pre_process(images, training):
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images


"""
This function creates the main part of the CNN
"""
def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # We use batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss


"""
This function creates the full neural network, which consists of the
pre-processing and main-processing
"""
def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x
        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)
        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


"""
Retrieve an existing variable named 'weights' in the scope with the given layer_name.
"""
def get_weights_variable(layer_name):
    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


"""
Retrieve the outputs of the convolutional layers
"""
def get_layer_output(layer_name):
    tensor_name = "network/" + layer_name + "/Relu:0"

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor


"""
Function for selecting a random batch of images from the training set
"""
def random_batch():
    # Number of images in the training-set.
    num_images = len(data["train"]["images"])

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = data["train"]["images"][idx, :, :, :]
    y_batch = data["train"]["labels"][idx, :]

    return x_batch, y_batch


"""
This function performs a number of optimization iterations to improve
the variables of the network layers. In each iteration, a new batch of data
is selected from the training set.
The progress is printed every 100 iterations.
A checkpoint is saved every 500 iterations.
"""
def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 500 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a checkpoint.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


"""
This function calculates the predicted classes of images and also returns 
a boolean array whether the classification of each image is correct.
"""
def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    return correct, cls_pred
    

"""
This function calculates the predicted class for the test-set
"""
def predict_cls_test():
    return predict_cls(images = data["test"]["images"],
                       labels = data["test"]["labels"],
                       cls_true = data["test"]["cls"])


"""
This function calculates the classification accuracy given 
a boolean array whether each image was correctly classified
"""
def classification_accuracy(correct):
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    
    return correct.mean(), correct.sum()
    

"""
Function for printing the classification accuracy on the test-set
"""
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()
    
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_functions.plot_example_errors(cls_pred=cls_pred, correct=correct, db=data["test"])

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_functions.plot_confusion_matrix(cls_pred=cls_pred, cls_true=data["test"]["cls"],
                                             num_classes=num_classes, class_names=class_names)
        
        
########################################################
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

########################################################
# SAVER to save the variables of the NN
saver = tf.train.Saver()
save_dir = 'checkpoints/'
save_path = os.path.join(save_dir, 'cifar10_cnn')

########################################################

def main(_):
    global data
    global class_names
    global session
    
    class_names, data = load_cifar10.load_data()
    print(class_names)
    session = tf.Session()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        print("Trying to restore last checkpoint ...")
    
        # Use TensorFlow to find the latest checkpoint - if any.
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    
        # Try and load the data in the checkpoint.
        saver.restore(session, save_path=last_chk_path)
    
        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_chk_path)
    except:
        # If the above failed for some reason, simply
        # initialize all the variables for the TensorFlow graph.
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())
    
    optimize(num_iterations=2500)
    print_test_accuracy(show_confusion_matrix=True)



if __name__ == "__main__":
    tf.app.run()