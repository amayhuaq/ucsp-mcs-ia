"""
@author: Angela Mayhua
"""
import matplotlib.pyplot as plt
import numpy as np
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2

path_basenet = 'caffenet/'
path_mydata = 'db/'
path_savemodel = 'revision/'
num_fold = 'F1'
NUM_CLASSES = 10
typeNet = None


# Execution in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

# Loading weights of pretrained net
import os
weights_pretrained = path_basenet + 'bvlc_reference_caffenet.caffemodel'
assert os.path.exists(weights_pretrained)

# Load ImageNet labels to imagenet_labels
imagenet_label_file = path_basenet + 'synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))

# Load chart labels to chart_labels
chart_label_file = path_mydata + 'chart_names.txt'
chart_names = list(np.loadtxt(chart_label_file, str, delimiter='\n'))


"""
Helper function for deprocessing preprocessed images, e.g., for display.
"""
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

"""
Take an array of shape (n, height, width) or (n, height, width, 3)
and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
"""
def vis_square(data):
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.figure()
    plt.imshow(data); plt.axis('off')

"""
Save the net structure into an image
"""
def draw_net(net_filename, output):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_filename).read(), net)
    caffe.draw.draw_net_to_file(net, output)


from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

# Helper functions to create layers
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


"""
Returns a NetSpec specifying CaffeNet, following the original proto text
specification (./models/bvlc_reference_caffenet/train_val.prototxt).
"""
def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    
    mode = 'train' if train else 'test'
    filename = path_savemodel + num_fold + '_' + typeNet + '_' + mode + '.prototxt'
    with open(filename, 'w') as f:
        f.write(str(n.to_proto()))
        return filename

"""
Create net to chart classification using a txt file (for training or testing).
Each line of this file has the image path and its label
"""
def chart_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'test'
    source = path_mydata + num_fold + '_%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=227, mean_file=path_basenet + 'imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_CLASSES,
                    classifier_name='fc8_revision',
                    learn_all=learn_all)

"""
Display predictions for an input image, it shows the k most probable classes and its corresponding percentage
"""
def disp_preds(net, image, labels, k=10, name='ImageNet'):
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    print '\nPredicted label: ', labels[probs.argmax()]
    
    top_k = (-probs).argsort()[:k]
    print 'Top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))
    
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0,2,3,1))
    
    feat = net.blobs['conv1'].data[0,:36]
    vis_square(feat)

"""
Display predictions for an ImageNet net, using imagenet labels
"""
def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

"""
Display predictions for a ChartNet classifier, using chart names as labels
"""
def disp_chart_preds(net, image, name='ChartNet'):
    disp_preds(net, image, chart_names, name=name)

"""
Create solver to train the network. 
It has the learning and configuration parameters
"""
def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    s.iter_size = 1
    s.max_iter = 100000     # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 100 iterations.
    s.display = 100

    # Snapshots are files used to store networks we've trained.
    s.snapshot = 500
    s.snapshot_prefix = 'revision/' + num_fold + '_finetune_chart'
    
    # Train on the GPU.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    filename = path_savemodel + num_fold + '_' + typeNet + '_solver.prototxt'
    with open(filename, 'w') as f:
         f.write(str(s))
         return filename

"""
Run solvers for niter iterations, returning the loss and accuracy recorded each iteration.
`solvers` is a list of (name, solver) tuples.
We invoke the solver to train the chart classification layer
"""
def run_solvers(niter, solvers, disp_interval=10):    
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers} for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy() for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100 * acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weights = {}
    for name, s in solvers:
        filename = path_savemodel + num_fold + '_weights.%s.caffemodel' % name
        weights[name] = filename
        s.net.save(weights[name])
    return loss, acc, weights

"""
Evaluation of the chart net using the weights of the training
"""
def eval_chart_net(weights, test_iters=10):
    test_net = caffe.Net(chart_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy

"""
Return the class of the input image using the net
"""
def predict_class_image(net, path_image):
    # Mean BGR
    mu = np.array([104, 117, 123])
    # Create transformer to data layer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # Move channel of image to the major dimension (R channel)
    transformer.set_transpose('data', (2,0,1))
    # Substraction of the mean over the image
    transformer.set_mean('data', mu)
    # Rescale of [0,1] a [0,255]
    transformer.set_raw_scale('data', 255)
    # Swap channels from RGB to BGR
    transformer.set_channel_swap('data', (2,1,0))
    # Reading image to classify
    image = caffe.io.load_image(path_image)
    # Get class of image
    net.blobs['data'].data[0, ...] = transformer.preprocess('data', image)
    probs = net.forward(start='conv1')['probs'][0]
    return probs.argmax()
    
"""
Classify the image using the net
"""
def classify_image(net, path_image):
    # Mean BGR
    mu = np.array([104, 117, 123])
    
    # Create transformer to data layer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # Move channel of image to the major dimension (R channel)
    transformer.set_transpose('data', (2,0,1))
    # Substraction of the mean over the image
    transformer.set_mean('data', mu)
    # Rescale of [0,1] a [0,255]
    transformer.set_raw_scale('data', 255)
    # Swap channels from RGB to BGR
    transformer.set_channel_swap('data', (2,1,0))
    # Reading image to classify
    image = caffe.io.load_image(path_image)
    # Show results of classification
    plt.figure()
    plt.imshow(image)
    disp_preds(net, transformer.preprocess('data', image), chart_names, name='ChartNet')    

"""
Evaluation of net per class, this function returns the accuracy percentage per class
"""
def eval_chartnet_per_class(net_filename, weights, test_file):
    test_net = caffe.Net(net_filename, weights, caffe.TEST)
    test_net.forward()
    accuracy = np.array([0 for i in range(NUM_CLASSES)])
    cont = np.array([0 for i in range(NUM_CLASSES)])
    
    with open(test_file) as f:
        content = f.readlines()
    content = [x.split( ) for x in content]
    
    for i in xrange(len(content)):
        pred_cls = predict_class_image(test_net, content[i][0])
        real_cls = int(content[i][1])
        cont[real_cls] += 1
        #print '\nPredicted: %s, Real: %s' %(chart_names[pred_cls], chart_names[real_cls])
        if pred_cls == real_cls:
            accuracy[pred_cls] += 1      
    
    for cls in xrange(NUM_CLASSES):
        accuracy[cls] = accuracy[cls] * 100.0 / cont[cls]
        print '\nClass %s = %5.2f' % (chart_names[cls], accuracy[cls])
    
    return test_net, accuracy
    
"""
Show information at the begining of the network (without training),
each class has the probability of 1 / NUM_CLASSES * 100%
"""
def show_untrained_net_info():
    global typeNet
    typeNet = 'imagenet'
    # Load pretrained net 
    dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
    imagenet_net_filename = caffenet(data=dummy_data, train=False)
    imagenet_net = caffe.Net(imagenet_net_filename, weights_pretrained, caffe.TEST)
    
    typeNet = 'untrained'
    # Untrained net to chart classification
    untrained_net_filename = chart_net(train=False, subset='train')
    untrained_chart_net = caffe.Net(untrained_net_filename, weights_pretrained, caffe.TEST)
    untrained_chart_net.forward()
    chart_data_batch = untrained_chart_net.blobs['data'].data.copy()
    chart_label_batch = np.array(untrained_chart_net.blobs['label'].data, dtype=np.int32)
    
    # image sample to show distribution per class
    plt.figure()
    batch_index = 10
    image = chart_data_batch[batch_index]
    plt.imshow(deprocess_net_image(image))
    print 'Actual label = ', chart_names[chart_label_batch[batch_index]]
    
    # show probabilities per class in the two nets
    disp_imagenet_preds(imagenet_net, image)
    disp_chart_preds(untrained_chart_net, image, name='ChartNet Untrained')
    
    draw_net(untrained_net_filename, "chartnet.jpg")
    del untrained_chart_net, imagenet_net

"""
We create two solvers: one uses the weights of a pretrained net (CaffeNet) 
and the second net uses random values.
"""
def train_and_test(niter=200, learn_all_pre=False, learn_all_scra=False):
    global typeNet
    typeNet = 'pretrained_net'
    # Initialize chartnet with pretrained ImageNet weights.
    chart_solver = caffe.get_solver(solver(chart_net(train=True, learn_all=learn_all_pre)))
    chart_solver.net.copy_from(weights_pretrained)
    
    typeNet = 'scratch_net'
    # Create chartnet that is initialized randomly.
    scratch_chart_solver = caffe.get_solver(solver(chart_net(train=True, learn_all=learn_all_scra)))
    
    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained', chart_solver), ('scratch', scratch_chart_solver)]
    loss, acc, weights = run_solvers(niter, solvers, disp_interval=50)
    print 'Done.'
    
    train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
    train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
    chart_weights, scratch_chart_weights = weights['pretrained'], weights['scratch']
    
    # Delete solvers to save memory.
    del chart_solver, scratch_chart_solver, solvers
    
    plt.figure()
    plt.plot(np.vstack([train_loss, scratch_train_loss]).T)
    plt.xlabel('Iteration #')
    plt.ylabel('Loss')
    plt.figure()
    plt.plot(np.vstack([train_acc, scratch_train_acc]).T)
    plt.xlabel('Iteration #')
    plt.ylabel('Accuracy')
    
    typeNet = 'pretrained_net'
    test_net, accuracy = eval_chart_net(chart_weights)
    print 'Accuracy, trained from ImageNet initialization: %3.1f%%' % (100 * accuracy, )
    typeNet = 'scratch_net'
    scratch_test_net, scratch_accuracy = eval_chart_net(scratch_chart_weights)
    print 'Accuracy, trained from   random initialization: %3.1f%%' % (100 * scratch_accuracy, )
    
    classify_image(test_net, '00024.png')
    classify_image(scratch_test_net, '00024.png')
    
    return chart_weights, scratch_chart_weights
        
    
def main():
    niter = 2000
    show_untrained_net_info()
    train_and_test(niter, learn_all_scra=True)
    
    print 'Evaluation pretrained net:\n'
    eval_chartnet_per_class(path_savemodel + num_fold + '_pretrained_net_test.prototxt', 
                            path_savemodel + num_fold + '_weights.pretrained.caffemodel',
                            path_mydata + num_fold + '_test.txt')
    print 'Evaluation scratch net:\n'
    eval_chartnet_per_class(path_savemodel + num_fold + '_scratch_net_test.prototxt', 
                            path_savemodel + num_fold + '_weights.scratch.caffemodel',
                            path_mydata + num_fold + '_test.txt')
    
    
if __name__ == "__main__":
    main()