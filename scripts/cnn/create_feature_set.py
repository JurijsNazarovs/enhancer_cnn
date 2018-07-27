import argparse  # to take arguments form terminal
import numpy as np
import random


def print_line():
    print("------------------------------------------------")


def one_hot(read, is_shuffle):

    one_hot = []
    chars = list(read.lower())  # create an array of characters
    if is_shuffle == 1:
        chars = list(np.random.permutation(chars[0:len(chars) - 1]))
        chars.append('\n')

    for i in chars:
        if i == 'a':
            one_hot.extend([1, 0, 0, 0])
        elif i == 'c':
            one_hot.extend([0, 1, 0, 0])
        elif i == 't':
            one_hot.extend([0, 0, 1, 0])
        elif i == 'g':
            one_hot.extend([0, 0, 0, 1])
        # else:
        #    one_hot.extend([0, 0, 0, 0])

    return one_hot


def handle_sample(fi, classLabel, is_shuffle=0):
    featureset = []
    try:
        with open(fi, 'r') as f:
            lines = f.readlines()
            for line in lines:
                one_hot_tmp = one_hot(line, is_shuffle)
                if len(one_hot_tmp) != 0:
                    featureset.append([one_hot_tmp, classLabel])
    except IOError:
        print("Cannot process the file: ",  fi)
        exit(1)

    return featureset


def create_features_and_labels(pos_file, neg_file, train_size, valid_size):
    features = []
    features += handle_sample(pos_file, [1, 0])
    is_shuffle = 0
    if (pos_file == neg_file):
        is_shuffle = 1
    features += handle_sample(neg_file, [0, 1], is_shuffle)

    random.shuffle(features)

    features = np.array(features)
    train_end = int(train_size * len(features))
    valid_end = int(valid_size * len(features)) + train_end
    train_x = list(features[:, 0][:train_end])
    train_y = list(features[:, 1][:train_end])

    valid_x = list(features[:, 0][train_end:valid_end])
    valid_y = list(features[:, 1][train_end:valid_end])

    test_x = list(features[:, 0][valid_end:])
    test_y = list(features[:, 1][valid_end:])
    return train_x, train_y, valid_x, valid_y, test_x, test_y


# Executed part
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for CNN")
    parser.add_argument('pos', type=str, metavar='1.',
                        help='file with positive samples')
    parser.add_argument('neg', type=str, metavar='2.',
                        help='file with negative samples')
    parser.add_argument('--train_size', type=float, metavar='',
                        default=0.7, help='train_size')
    parser.add_argument('--validation_size', type=float, metavar='',
                        default=0.2, help='validation_size')
    try:
        args = parser.parse_args()
    except SystemExit as err:
        if err.code == 2:
            parser.print_help()
        exit(0)

    print_line()
    print("Prepare data for cnn:\n",
          "------------------------------------------------\n",
          "Positive sample file:   ", args.pos, "\n",
          "Negative sample file:   ", args.neg, "\n",
          "Training sample size:   ", args.train_size, "\n",
          "Validation sample size: ", args.validation_size)
    print_line()

    # Check validation of sizes
    if args.train_size == 0:
        print("Error: train_size cannot be 0")
        exit(1)

    for x in [args.train_size, args.validation_size]:
        if (x < 0 or x > 1):
            print("Error: size has to be in a range (0, 1]")
            exit(1)

    if not (args.train_size + args.validation_size > 0 and
            args.train_size + args.validation_size <= 1):
        print("Error: train_size + validation_size has to be in a range (0,1)")
        exit(1)

    # Create set
    train_x, train_y, valid_x, valid_y, test_x, test_y = create_features_and_labels(
        args.pos, args.neg, args.train_size, args.validation_size)

    # Save as a pickle file
    print_line()
    print("Saving as pickle file:")

    import pickle
    import os
    file_out_name = os.path.dirname(os.path.abspath(
        args.pos + "/..")) + "/pyData/"

    os.makedirs(file_out_name, exist_ok=True)

    file_out_name += \
        os.path.splitext(os.path.basename(args.pos))[0] + ".cnn_data"

    if args.pos == args.neg:
        file_out_name += ".neg_is_shuffle"

    file_out_name += ".pickle"

    with open(file_out_name, 'wb') as f:
        pickle.dump([train_x, train_y, valid_x, valid_y, test_x, test_y], f)
    print(file_out_name + " ... done!")
    print_line()
