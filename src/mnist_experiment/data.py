"""
Data
~~~~
Create lists of images for training and testing, along with lists of answers from a given csv file. The csv file format is from the MNIST data set with rows of data like this:
'7,0,0,70,181,255,...,0,0'
The first entry on each line is the label (number from 0 to 9) and the remaining entries are the pixel values for the image. Max pixel value is 255. Everything is rescaled so that inputs to the initial layer will be strictly between 0 and 1.

Member methods:
* __init__
* import_and_format
"""

# import numpy as np
import os
from numpy import identity, zeros


# The number of different target outputs (ie the digits from 0 to 9).
NUM_TARGETS = 10


class Data:
    def __init__(
        self,
        directory: str,
        data_file_loc: str,
    ) -> None:
        self.data_path = os.path.join(directory, data_file_loc)
        self.num_targets = NUM_TARGETS

        # These are the target output values. Each row encodes a number
        # from 0 to 9 as a unit vector of the form:
        # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] -> represents '2'
        self.targets = (identity(self.num_targets) * 0.98 + 0.01)
        self.import_and_format()
        return

    def import_and_format(
        self,
    ) -> None:
        with open(self.data_path) as data_file:
            data_strings = data_file.read().split("\n")
            self.num_images = len(data_strings)
            data_lists = [
                data_strings[i].split(",")
                for i in range(self.num_images)
            ]
            self.num_pixels = len(data_lists[0]) - 1
            self.images = zeros((self.num_images, self.num_pixels))
            for i in range(self.num_images):
                for j in range(1, self.num_pixels + 1):
                    pixel_value = float(data_lists[i][j])
                    self.images[i][j - 1] = pixel_value / 255.0 * 0.99 + 0.01
            self.answers = zeros(self.num_images).reshape(-1, 1)
            for i in range(self.num_images):
                self.answers[i] = float(data_lists[i][0])

        # Encode answers (digits from 0 to 9) as corresponding unit
        # vector from self.targets
        self.encoded_answers = [
            self.targets[int(answer)]
            for answer in self.answers
        ]
        return
