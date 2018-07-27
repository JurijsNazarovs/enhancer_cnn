import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import tensorflow as tf
import os

# Explanation of STRIDES (which means iteration):
# A 4-dim tensor has shape [batch, height, width, channel]. Numbers in
# strides correspond to these values. 1-st and 4-th has to be 1 by
# Tensorflow design. 2nd and 3rd might be changed if there is a need.
# Better explanation is here: https://github.com/Hvass-Labs/TensorFlow-Tutorials/issues/19


# 2-dimensional conv. Cant get depth here
def conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


def maxpool2d(x, height, width):  # size - size of matrix to pool
    return tf.nn.max_pool(x, ksize=[1, height, width, 1],
                          strides=[1, height, width,  1], padding='SAME')
# Use same hight and with in ksize and strides to avoid overlapping of matrices


# [Start] Generated code
def convolutional_neural_network(x, x_h, x_w,
                                 conv_matrix_h, padding='VALID', features_type='TF',
                                 conv_n_filters=-1,
                                 pool_h=2, pool_w=1, n_fc=200, n_classes=2):
    # Parameters to define the structure of CNN
    if (conv_n_filters <= 0):
        conv_n_filters = min(x_w ** conv_matrix_h, 32)

    if (x_h < conv_matrix_h):
        print("Error: conv_matrix_h cannot be less than size of image")
        exit(1)

    strides = [1, 1, 1, 1]
    if (padding == 'VALID'):
        out_h = math.ceil(float(x_h - conv_matrix_h + 1) / float(strides[1]))
        out_w = math.ceil(float(x_w - x_w + 1) / float(strides[2]))
    else:
        out_h = x_h
        out_w = x_w

    weights = {
        'W_fc': tf.Variable(tf.random_normal([math.ceil(out_h / pool_h) *
                                              math.ceil(out_w / pool_w) *
                                              conv_n_filters,
                                              n_fc])),
        'out': tf.Variable(tf.random_normal([n_fc, n_classes]))
    }

    # Define features for convolultion layer
    # height, width, input - # of colors, output - # of filters
    if features_type.lower() == "tf":
        weights.update({'W_conv1': tf.Variable(tf.random_normal([conv_matrix_h,
                                                                 x_w,
                                                                 1,
                                                                 conv_n_filters]))})
    elif features_type.lower() == "k-mer":
        weightsTmp = generate_k_mers(conv_matrix_h, conv_n_filters)
        weights.update({'W_conv1': tf.Variable(tf.reshape(weightsTmp,
                                                          [conv_matrix_h,
                                                           x_w,
                                                           1,
                                                           conv_n_filters]),
                                               trainable=False)})
    else:
        print("Features type is not supported")
        exit(1)

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([conv_n_filters])),
        'b_fc': tf.Variable(tf.random_normal([n_fc])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to a 4D tensor: [batch, height, width, channel]
    x = tf.reshape(x, shape=[-1, x_h, x_w, 1])

    # ConvLayer 1
    conv1 = tf.nn.relu(
        conv2d(x, weights['W_conv1'], strides, padding) + biases['b_conv1'])
    conv1MaxPool = maxpool2d(conv1, pool_h, pool_w)

    # Fully connected layer
    # reshape to fit prev layer
    fc = tf.reshape(conv1MaxPool, [-1, math.ceil(out_h / pool_h) *
                                   math.ceil(out_w / pool_w) * conv_n_filters])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output,  weights["W_conv1"], (conv1, conv1MaxPool, fc), ("Convolution layer",
                                                                    "Max Pooling",
                                                                    "Fully Connected")
# [End] Generated code


# [Start] Generated code
def convolutional_neural_network_2(x, x_h, x_w,
                                   conv_matrix_h1,
                                   conv_matrix_h2,
                                   conv_n_filters1=-1,
                                   conv_n_filters2=-1,
                                   pool_h=60, pool_w=2, n_fc=20, n_classes=2):
    # Parameters to define the structure of CNN
    if (conv_n_filters1 <= 0):
        conv_n_filters1 = 32
    if (conv_n_filters2 <= 0):
        conv_n_filters2 = 64

    weights = {
        # height, width, input (number of colors), output (# of filters)
        'W_conv1': tf.Variable(tf.random_normal([conv_matrix_h1, x_w,
                                                 1, conv_n_filters1])),
        'W_conv2': tf.Variable(tf.random_normal([conv_matrix_h2, x_w,
                                                 conv_n_filters1,
                                                 conv_n_filters2])),
        'W_fc': tf.Variable(tf.random_normal([math.ceil(x_h / (pool_h ** 2)) *
                                              math.ceil(x_w / (pool_w ** 2)) *
                                              conv_n_filters2,
                                              n_fc])),
        'out': tf.Variable(tf.random_normal([n_fc, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([conv_n_filters1])),
        'b_conv2': tf.Variable(tf.random_normal([conv_n_filters2])),
        'b_fc': tf.Variable(tf.random_normal([n_fc])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to a 4D tensor: [batch, height, width, channel]
    x = tf.reshape(x, shape=[-1, x_h, x_w, 1])

    # ConvLayer 1
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1, pool_h, pool_w)

    # ConvLayer 2
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2, pool_h, pool_w)

    # Fully connected layer
    # reshape to fit prev layer
    fc = tf.reshape(conv2, [-1, math.ceil(x_h / (pool_h ** 2)) *
                            math.ceil(x_w / (pool_w ** 2)) *
                            conv_n_filters2])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output
# [End] Generated code


def train_neural_network(train_x, train_y, test_x, test_y,
                         conv_matrix_h=2,
                         n_iter=100,
                         n_epochs=10,
                         n_layers=1,
                         display_progress=1,
                         plot_network=0,
                         padding='VALID',
                         features_type='tf'):
    # Function train CNN and return an accuracy

    # Data Description:
    # Data presents a list of separated nucleotide sequence with
    # one-vector encoding. => every element of a list is of size:
    # sequenceSize * 4. 4 - A, C, T, G. Each is represented by
    # 1x4 vector of 0 and 1. Thus:
    # x_width - number of columns in pic representation because of hotencoding
    # x_height - number of rows in pic representation - length of sequence

    x_width = 4
    x_height = len(train_x[0]) // x_width  # integer division
    n_classes = 2  # classes: + and -

    batch_size = round(len(train_x) / n_iter)

    print_line()
    print("CNN summary:\n",
          "------------------------------------------------\n",
          "Training sample size:      ", len(train_x), "\n",
          "Number of positive labels: ", train_y.count([1, 0]), '\n',
          "Number of negative labels: ", train_y.count([0, 1]), '\n',
          "------------------------------------------------\n",
          "Testing sample size:       ", len(test_x), "\n",
          "Number of positive labels: ", test_y.count([1, 0]), '\n',
          "Number of negative labels: ", test_y.count([0, 1]), '\n',
          "------------------------------------------------\n",
          "Size of image:             ", x_height, "x", x_width, "\n",
          "------------------------------------------------\n",
          "Number of iterations:      ", n_iter, "\n",
          "Batch size:                ", batch_size, "\n",
          "Number of epoch:           ", n_epochs, "\n",
          "Conv matrix size:          ", conv_matrix_h, "x", x_width, "\n",
          "------------------------------------------------\n",
          "Features generator:        ", features_type, "\n",
          "Padding:                   ", padding)
    print_line()

    # CNN description
    # shape = height x width. Our height = none, we squiz data in a row of pixels
    x = tf.placeholder(tf.float32, shape=[None, len(train_x[0])])
    y = tf.placeholder(tf.int32, shape=[None, n_classes])

    if n_layers == 2:
        # not finished
        prediction = convolutional_neural_network_2(x, x_height, x_width,
                                                    conv_matrix_h,
                                                    conv_matrix_h)
    else:
        prediction, weights, layers, layersName = convolutional_neural_network(x,
                                                                               x_height,
                                                                               x_width,
                                                                               conv_matrix_h,
                                                                               padding,
                                                                               features_type,
                                                                               pool_h=2,
                                                                               pool_w=2)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(
        cost)  # here can be different optimizers

    # AUC
    auc_estim = tf.contrib.metrics.streaming_auc(tf.cast(tf.argmax(prediction, 1), tf.float32),
                                                 tf.cast(tf.argmax(y, 1), tf.float32))

    # CNN training
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)
        # Train
        for epoch in range(n_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={
                    x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

                # AUC
                auc = sess.run(
                    auc_estim, feed_dict={x: batch_x, y: batch_y})

            if display_progress == 1:
                print('Epoch', epoch + 1, 'completed out of',
                      n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # Calculate accuracy
        accuracy_estim = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy = accuracy_estim.eval({x: test_x, y: test_y})
        print('Accuracy: ', accuracy)
        print('AUC:', auc)

        # Plots
        if plot_network == 1:
            print_line()
            imageToPlot = train_x[0]
            # Original image
            plot_layer(np.reshape(imageToPlot, [
                len(imageToPlot) // 4, 4]), "Original Image")
            # Layers
            for layerIter in range(len(layers)):
                get_activations(layers[layerIter], layersName[layerIter],
                                imageToPlot, len(imageToPlot), sess, x)
            # Weights
            weights = sess.run(weights)
            weights = np.transpose(weights, (2, 0, 1, 3))
            plot_layer(weights, "Convolution Weights")

            print_line()
    return accuracy, auc[0]


# Generate features
def generate_k_mers(height, n):
    weights = np.full((height, 4, 1, n), 0, dtype=np.float32)
    for i in range(n):
        read = list(np.random.choice(['a', 'c', 't', 'g'], height))
        weights[:, :, 0, i] = one_hot(read)

    return (weights)


def one_hot(read):
    n = len(read)
    one_hot = np.full((n, 4), 0)
    for j in range(n):
        i = read[j]
        if i == 'a':
            one_hot[j, :] = ([1, 0, 0, 0])
        elif i == 'c':
            one_hot[j, :] = ([0, 1, 0, 0])
        elif i == 't':
            one_hot[j, :] = ([0, 0, 1, 0])
        elif i == 'g':
            one_hot[j, :] = ([0, 0, 0, 1])

    return one_hot


# Plots
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt


def get_activations(layer, layerName, inputImage, x_size, sess, x):
    units = sess.run(layer, feed_dict={x: np.reshape(
        inputImage, [1, x_size], order='F')})

    plot_layer(units, layerName)


def plot_layer(units, layerName):
    font_size = 30
    plt.figure(1, figsize=(20, 20))

    if len(units.shape) == 4:
        filters = units.shape[3]
        n_cols = 2
        n_rows = math.ceil(filters / n_cols) + 1

        if (units.shape[1] > units.shape[2]):
            units = np.transpose(units, (0, 2, 1, 3))
        for i in range(filters):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.title('Filter ' + str(i + 1), fontsize=font_size)
            plt.imshow(units[0, :, :, i],
                       interpolation="nearest", cmap="binary")

        plt.suptitle(layerName, fontsize=font_size + 10)
    else:
        plt.title(layerName, fontsize=font_size)
        plt.imshow(units, interpolation="nearest", cmap="binary")

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/" + layerName.replace(" ", "_") +
                ".pdf", bbox_inches='tight')
    plt.close()
    print(layerName + ' - plotted')


def print_line():
    print("------------------------------------------------")


if __name__ == '__main__':
    import argparse  # to take arguments form terminal
    parser = argparse.ArgumentParser(description="Train CNN")
    parser.add_argument('data_pickle', type=str, metavar='1.',
                        default="../pyData/pos.pickle",
                        help='pickle file with whole data set: ' +
                        'train_x, train_y, valid_x, valid_y, test_x, test_y')
    parser.add_argument('--conv_matrix_h', type=int, metavar='',
                        default=2,
                        help='height of convolution layer matrix')
    parser.add_argument('--n_iter', type=int, metavar='',
                        default=100,
                        help='number of iterations in one epoch')
    parser.add_argument('--n_epochs', type=int, metavar='',
                        default=10,
                        help='number of epochs')
    parser.add_argument('--features_type', type=str, metavar='',
                        default='tf',
                        help='Type of features: ' +
                        'tf - adaptive Tensorflow, k-mer - generate k-mer',
                        choices=['tf', 'k-mer'])
    parser.add_argument('--padding', type=str, metavar='',
                        default='VALID',
                        help='padding of convolution layer: ' +
                        'VALID, SAME', choices=['VALID', 'SAME'])
    parser.add_argument('--n_layers', type=int, metavar='',
                        default=2,
                        help='number of convolutino layers to run')

    parser.add_argument('--display_progress', type=int, metavar='',
                        default=1,
                        help='display progress of cnn training: ' +
                        '1 - yes, 0 - no', choices=[1, 0])
    parser.add_argument('--run_all', type=int, metavar='',
                        default=0,
                        help='run all possible nets: ' +
                        '1 - yes, 0 - no', choices=[1, 0])
    parser.add_argument('--plot_network', type=int, metavar='',
                        default=0,
                        help='plot all layers of a network ' +
                        '1 - yes, 0 - no', choices=[1, 0])
    parser.add_argument('--summary_file', type=str, metavar='',
                        default="cnnSummary.csv",
                        help='file to summarize results')
    try:
        args = parser.parse_args()
    except SystemExit as err:
        if err.code == 2:
            parser.print_help()
        exit(0)

    print_line()
    print("Data set: ", args.data_pickle)
    print("Loading ... ", end="")

    # Read data from pickle file
    import pickle
    with open(args.data_pickle, 'rb') as fi:
        train_x, train_y, valid_x, valid_y, test_x, test_y = pickle.load(fi)
    print("done!")
    print_line()
    # Train cnn
    if (args.run_all == 1):
        with open(args.summary_file, 'w', 1) as fi:
            for i in range(1, args.conv_matrix_h + 1):
                print_line
                print("Iteration: ", i, '/', args.conv_matrix_h)
                accuracy, auc = (train_neural_network(train_x, train_y,
                                                      valid_x, valid_y,
                                                      i, args.n_iter,
                                                      args.n_epochs,
                                                      args.n_layers,
                                                      args.display_progress,
                                                      args.plot_network,
                                                      args.padding,
                                                      args.features_type))
                print(i, accuracy, auc, sep=',', file=fi)
    else:
        accuracy = (train_neural_network(train_x, train_y, valid_x, valid_y,
                                         args.conv_matrix_h, args.n_iter,
                                         args.n_epochs,
                                         args.n_layers,
                                         args.display_progress,
                                         args.plot_network,
                                         args.padding,
                                         args.features_type))
        print('Accuracy:',  accuracy)
