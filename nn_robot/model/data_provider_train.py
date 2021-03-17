import numpy as np
import os
from PIL import Image
from util import image2tanh
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tfr_utilities_train import prepare_dataset

class data_provider():
    def __init__(self, dataset_rooth, epochs, batch_size, buffer_size, augment):
        self.batch_size = batch_size
        self.buffer_size_d = buffer_size
        self.augment = augment
        self._load_dataset(dataset_rooth)
        self.epochs = epochs

    def get_sample(self):
        return self.next_train, self.init_train

    def _load_dataset(self, dataset_rooth):
        self.next_train, self.init_train = prepare_dataset(dataset_rooth, self.batch_size,self.buffer_size_d,self.augment, None)

