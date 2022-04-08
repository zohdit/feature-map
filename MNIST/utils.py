import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging as log
import sys
# For Python 3.6 we use the base keras
import keras
# from tensorflow import keras
from config import BITMAP_THRESHOLD
import numpy as np

# local imports

IMG_SIZE = 28

import math
from sklearn.linear_model import LinearRegression
import re
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np

NAMESPACE = '{http://www.w3.org/2000/svg}'


def feature_simulator(function, x):
    """
    Calculates the value of the desired feature
    :param function: name of the method to compute the feature value
    :param x: genotype of candidate solution x
    :return: feature value
    """
    if function == 'bitmap_count':
        return bitmap_count(x, BITMAP_THRESHOLD)
    if function == 'move_distance':
        return move_distance(x)
    if function == 'orientation_calc':
        return orientation_calc(x, 0)


def bitmap_count(digit, threshold, normalize=False):
    image = np.asarray(digit.purified)
    if normalize:
        image = image / 255.
    return len(image[image > threshold])


def move_distance(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('([\d\.]+),([\d\.]+)\sM\s([\d\.]+),([\d\.]+)')
    segments = pattern.findall(svg_path)
    if len(segments) > 0:
        dists = []  # distances of moves
        for segment in segments:
            x1 = float(segment[0])
            y1 = float(segment[1])
            x2 = float(segment[2])
            y2 = float(segment[3])
            dist = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
            dists.append(dist)
        return int(np.sum(dists))
    else:
        return 0


def orientation_calc(digit, threshold):
    x = []
    y = []
    image = deepcopy(digit.purified)
    bw = np.asarray(image)
    for iz, ix, iy, ig in np.ndindex(bw.shape):
        if bw[iz, ix, iy, ig] > threshold:
            x.append([iy])
            y.append(ix)
    X = np.array(x)
    Y = np.array(y)
    lr = LinearRegression(fit_intercept=True).fit(X, Y)
    normalized_ori = -lr.coef_
    new_ori = normalized_ori * 100
    return int(new_ori)


def compute_sparseness(map, x):
    n = len(map)
    # Sparseness is evaluated only if the archive is not empty
    # Otherwise the sparseness is 1
    if (n == 0) or (n == 1):
        sparseness = 0
    else:
        sparseness = density(map, x)
    return sparseness


def get_neighbors(b):
    neighbors = []
    neighbors.append((b[0], b[1] + 1))
    neighbors.append((b[0] + 1, b[1] + 1))
    neighbors.append((b[0] - 1, b[1] + 1))
    neighbors.append((b[0] + 1, b[1]))
    neighbors.append((b[0] + 1, b[1] - 1))
    neighbors.append((b[0] - 1, b[1]))
    neighbors.append((b[0] - 1, b[1] - 1))
    neighbors.append((b[0], b[1] - 1))

    return neighbors


def density(map, x):
    b = x.features
    density = 0
    neighbors = get_neighbors(b)
    for neighbor in neighbors:
        if neighbor not in map:
            density += 1
    return density


def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0

    return x_reshape


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def print_image(filename, image, cmap=''):
    if cmap != '':
        plt.imsave(filename, image.reshape(28, 28), cmap=cmap, format='png')
    else:
        plt.imsave(filename, image.reshape(28, 28), format='png')
    np.save(filename, image)


# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v


def setup_logging(log_to, debug):
    def log_exception(extype, value, trace):
        log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append(file_handler)
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    sys.excepthook = log_exception

    log.info(start_msg)
