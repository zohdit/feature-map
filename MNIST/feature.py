import numpy as np
from sample import Sample
import utils as us

class Feature:
    """
    Implements a feature dimension of the Feature map
    """

    def __init__(self, feature_name, feature_simulator):
        """
        :param feature_name: Name of the feature dimension
        :param feature_simulator: Name of the method to evaluate the feature
        :param bins: Array of bins, from starting value to last value of last bin
        """
        self.feature_name = feature_name
        self.feature_simulator = feature_simulator
        self.min_value = np.inf
        self.max_value = 0


    def feature_descriptor(self, sample: Sample):
        """
        Simulate the candidate solution x and record its feature descriptor
        :param x: genotype of candidate solution x
        :return:
        """
        i = us.feature_simulator(self.feature_simulator, sample)
        return i
    
    def get_coordinate_for(self, sample: Sample):


        # TODO Check whether the sample has the feature
        value = sample.features[self.feature_name]


        return value

