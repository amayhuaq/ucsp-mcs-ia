import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""
Function used to plot 9 images in a 3x3 grid and show the true and predicted class
images      Array of images to plot
class_names Array of class_names for each image
cls_true    Array of true class for each image
cls_pred    Array of predicted class for each image
smooth      Boolean to smooth or not the image
"""
def plot_images(images, class_names, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :], interpolation=interpolation)
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # We show the figure.
    plt.show()


"""
Function used to plot images that were incorrectly classified.
cls_pred    Array of the predicted class-number for each image in db["data"]
correct     Boolean array, TRUE whether the predicted class is equal to the true class
db          Array with the images and labels. Ex: db = {"data":[], "labels":[], "cls":[]}
"""
def plot_example_errors(cls_pred, correct, db):
    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been incorrectly classified.
    images = db["images"][incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = db["cls"][incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


"""
Function used to plot the confusion matrix of the classifier
cls_pred    Array of the predicted class-number
cls_true    Array of the true class-number
num_classes Number of classes
class_names Array of class names
"""
def plot_confusion_matrix(cls_pred, cls_true, num_classes, class_names):
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
