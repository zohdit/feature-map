import math
import re
from enum import Enum
from xml.etree import ElementTree as et

import numpy as np
from sklearn.linear_model import LinearRegression

from config import BITMAP_THRESHOLD
from sample import Sample

NAMESPACE = '{http://www.w3.org/2000/svg}'


def bitmap_count(sample: Sample, threshold: float = BITMAP_THRESHOLD, normalize=False):
    image = np.asarray(sample.purified)
    if normalize:
        image = image / 255.
    return len(image[image > threshold])


def move_distance(sample: Sample):
    root = et.fromstring(sample.xml_desc)
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


def orientation_calc(sample: Sample, threshold: float = 0):
    # TODO: find out why x and y are inverted
    # convert the image to an array and remove the 1 dimensions (bw) -> 2D array
    image = np.squeeze(sample.purified)
    # get the indices where the matrix is greater than the threshold
    y, x = np.where(image > threshold)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(x.reshape(-1, 1), y)
    orientation = -lr.coef_[0] * 100
    return int(orientation)


class FeatureSimulator(Enum):
    BITMAP_COUNT = bitmap_count
    MOVE_DISTANCE = move_distance
    ORIENTATION_CALC = orientation_calc
