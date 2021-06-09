"""
For poking about and testing all the stuff.
"""

import numpy as np
# import random
# import nn_experiment.neural_network as nn

with open("/Users/jdodd/code/mnist/data/mnist_test.csv") as data_file:
    data_strings = data_file.read().split("\n")
    num_images = len(data_strings)
    data_lists = [
        data_strings[i].split(",")
        for i in range(num_images)
    ]
    num_pixels = len(data_lists[0]) - 1
    # images = np.ndarray(
    #     [
    #         int(data_lists[i][j])
    #         for i in range(num_images)
    #         for j in range(1, num_pixels + 1)
    #     ]
    # ).reshape(num_images, -1)
    # images_list = [
    #     int(data_lists[i][j])
    #     for i in range(num_images)
    #     for j in range(1, num_pixels + 1)
    # ]
    images_list = np.zeros((num_images, num_pixels))
    for i in range(num_images):
        for j in range(1, num_pixels + 1):
            images_list[i][j - 1] = int(data_lists[i][j])
    print(images_list[0])
#    images = np.ndarray(images_list)
