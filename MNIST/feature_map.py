import os
import time

import pandas as pd
import tensorflow as tf

from feature import Feature
from utils.feature_map.preprocess import extract_samples_and_stats
from utils.feature_map.visualize import visualize_map

DATA_FILE = 'logs/featuremaps_data'

if __name__ == "__main__":
    # Load the data
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    # Extract the samples and the stats
    start_time = time.time()
    samples, stats = extract_samples_and_stats(x_test, y_test)
    # Get the list of features
    features = [
        Feature(feature_name, feature_stats['min'], feature_stats['max'])
        for feature_name, feature_stats in stats.to_dict().items()
    ]
    features_extraction_time = time.time() - start_time
    # Visualize the feature-maps
    data = visualize_map(features, samples)
    for features_dict in data:
        features_dict.update({'features_extraction': features_extraction_time})

    # Update the current data or create a new dataframe
    if os.path.isfile(DATA_FILE):
        old_data = pd.read_pickle(DATA_FILE)
        new_data = pd.DataFrame(data)
        features_df = pd.concat([old_data, new_data]).drop_duplicates(subset='approach', keep='last')
    else:
        features_df = pd.DataFrame(data)
    features_df.to_pickle(DATA_FILE)
